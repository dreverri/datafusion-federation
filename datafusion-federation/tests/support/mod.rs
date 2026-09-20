//! Shared test helpers for the federation integration tests.
//!
//! A [`RecordingSQLExecutor`] plays the role of a "remote database": it forwards
//! the pushed-down SQL to a local DataFusion [`SessionContext`] (backed by a CSV,
//! exactly like the examples) and records every query string it receives. Tests
//! can then assert on that recorded SQL to prove that compute was pushed down to
//! the remote instead of executed locally.

#![cfg(feature = "sql")]
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::{
    arrow::{array::RecordBatch, datatypes::SchemaRef},
    catalog::SchemaProvider,
    execution::{
        context::{SessionContext, SessionState},
        options::CsvReadOptions,
        SessionStateBuilder,
    },
    physical_plan::{stream::RecordBatchStreamAdapter, PhysicalExpr, SendableRecordBatchStream},
    prelude::DataFrame,
    sql::unparser::dialect::{DefaultDialect, Dialect},
};
use futures::TryStreamExt;

use datafusion_federation::sql::{SQLExecutor, SQLFederationProvider, SQLSchemaProvider};

/// A CSV-backed [`SQLExecutor`] that records the SQL it is asked to execute.
///
/// The recorded queries are the exact strings federation unparsed and handed to
/// the remote engine, so asserting on them verifies what was pushed down.
pub struct RecordingSQLExecutor {
    name: &'static str,
    context: &'static str,
    session: Arc<SessionContext>,
    queries: Arc<Mutex<Vec<String>>>,
}

impl RecordingSQLExecutor {
    pub fn new(name: &'static str, context: &'static str, session: Arc<SessionContext>) -> Self {
        Self {
            name,
            context,
            session,
            queries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A handle to the list of SQL strings this executor has received.
    pub fn queries(&self) -> Arc<Mutex<Vec<String>>> {
        self.queries.clone()
    }
}

#[async_trait]
impl SQLExecutor for RecordingSQLExecutor {
    fn name(&self) -> &str {
        self.name
    }

    fn compute_context(&self) -> Option<String> {
        // A unique, stable context so distinct remotes are never merged.
        Some(self.context.to_string())
    }

    fn execute(
        &self,
        sql: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        self.queries.lock().unwrap().push(sql.to_string());

        let session = self.session.clone();
        let sql = sql.to_string();
        let future_stream = async move { session.sql(&sql).await?.execute_stream().await };
        let stream = futures::stream::once(future_stream).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    async fn table_names(&self) -> datafusion::error::Result<Vec<String>> {
        Err(datafusion::error::DataFusionError::NotImplemented(
            "table inference not implemented".to_string(),
        ))
    }

    async fn get_table_schema(&self, table_name: &str) -> datafusion::error::Result<SchemaRef> {
        let sql = format!("select * from {table_name} limit 1");
        let df = self.session.sql(&sql).await?;
        Ok(Arc::new(df.schema().as_arrow().clone()))
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        Arc::new(DefaultDialect {})
    }
}

/// Absolute path to a test CSV, resolved against the crate manifest dir so
/// tests work regardless of the working directory.
pub fn data_path(file: &str) -> String {
    format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"))
}

/// Build a "remote" [`SessionContext`] with `csv` registered as `table_name`.
pub async fn remote_ctx(table_name: &str, csv: &str) -> Arc<SessionContext> {
    let ctx = Arc::new(SessionContext::new());
    ctx.register_csv(table_name, data_path(csv), CsvReadOptions::new())
        .await
        .expect("register csv");
    ctx
}

/// Register `schema` as the default schema of `state`.
pub fn overwrite_default_schema(state: &SessionState, schema: Arc<dyn SchemaProvider>) {
    let options = &state.config().options().catalog;
    let catalog = state
        .catalog_list()
        .catalog(options.default_catalog.as_str())
        .expect("default catalog");
    catalog
        .register_schema(options.default_schema.as_str(), schema)
        .expect("register schema");
}

/// Wrap `executor` in a federated schema provider exposing `tables`.
pub async fn schema_provider(
    executor: Arc<dyn SQLExecutor>,
    tables: &[&str],
) -> Arc<SQLSchemaProvider> {
    let provider = Arc::new(SQLFederationProvider::new(executor));
    let tables: Vec<String> = tables.iter().map(|t| t.to_string()).collect();
    Arc::new(
        SQLSchemaProvider::new_with_tables(provider, tables)
            .await
            .expect("schema provider"),
    )
}

/// Run `query` against a federation-enabled context whose default schema is
/// `schema`, returning the collected result batches.
pub async fn run_federated(schema: Arc<dyn SchemaProvider>, query: &str) -> Vec<RecordBatch> {
    let state = datafusion_federation::default_session_state();
    overwrite_default_schema(&state, schema);
    let ctx = SessionContext::new_with_state(state);
    collect(ctx.sql(query).await.expect("plan query")).await
}

/// Collect a dataframe into batches.
pub async fn collect(df: DataFrame) -> Vec<RecordBatch> {
    df.collect().await.expect("collect")
}

/// Run `query` against a context whose default schema is `schema` but WITHOUT
/// the federation optimizer rule or query planner. This is the negative
/// control: the `FederatedTableProviderAdaptor` is never swapped for a
/// federation node, so scanning it fails. Returns the (expected) error.
pub async fn try_run_without_federation(
    schema: Arc<dyn SchemaProvider>,
    query: &str,
) -> datafusion::error::Result<Vec<RecordBatch>> {
    let state = SessionStateBuilder::new().with_default_features().build();
    overwrite_default_schema(&state, schema);
    let ctx = SessionContext::new_with_state(state);
    ctx.sql(query).await?.collect().await
}

/// Total number of rows across `batches`.
pub fn row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// The concatenation of all SQL strings recorded so far, lowercased for
/// case-insensitive matching.
pub fn recorded_sql(queries: &Arc<Mutex<Vec<String>>>) -> String {
    queries.lock().unwrap().join("\n").to_lowercase()
}

/// Register `schema` under `name` in the default catalog of `state`.
///
/// Unlike [`overwrite_default_schema`], this leaves room for more than
/// one, which is what a cross-engine test needs: two federated schemas
/// with different compute contexts, so a join between them cannot be
/// folded into a single remote query.
pub fn register_named_schema(state: &SessionState, name: &str, schema: Arc<dyn SchemaProvider>) {
    let options = &state.config().options().catalog;
    let catalog = state
        .catalog_list()
        .catalog(options.default_catalog.as_str())
        .expect("default catalog");
    catalog
        .register_schema(name, schema)
        .expect("register schema");
}

/// A federation-enabled context holding two independent "remote
/// databases", as schemas `alpha` and `beta`.
///
/// Each records the SQL it is sent, and the two report different compute
/// contexts, so federation must plan them as separate remote queries
/// joined locally.
pub async fn two_remotes() -> (
    SessionContext,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<String>>>,
) {
    let alpha_exec =
        RecordingSQLExecutor::new("alpha", "alpha_ctx", remote_ctx("test", "test.csv").await);
    let beta_exec =
        RecordingSQLExecutor::new("beta", "beta_ctx", remote_ctx("test2", "test2.csv").await);
    let (alpha_sql, beta_sql) = (alpha_exec.queries(), beta_exec.queries());

    let alpha = schema_provider(Arc::new(alpha_exec), &["test"]).await;
    let beta = schema_provider(Arc::new(beta_exec), &["test2"]).await;

    let state = datafusion_federation::default_session_state();
    register_named_schema(&state, "alpha", alpha);
    register_named_schema(&state, "beta", beta);

    (SessionContext::new_with_state(state), alpha_sql, beta_sql)
}

/// A context where the table is registered under a *different* name than the
/// remote knows it by — `local` here, `remote` there.
///
/// Every other helper registers each table under its own remote name, which
/// makes the analyzer's table-scan rewrite an identity and hides any bug in
/// it. A catalog that addresses tables by a generated id (`t42`) renames every
/// table it serves, so the rewrite is the normal case there, not the exotic
/// one.
pub async fn renamed_remote(
    local: &str,
    remote: &str,
    csv: &str,
) -> (SessionContext, Arc<Mutex<Vec<String>>>) {
    use datafusion::{
        arrow::datatypes::{DataType, Field, Schema},
        common::TableReference,
    };
    use datafusion_federation::{sql::SQLTableSource, FederatedTableProviderAdaptor};

    let executor =
        RecordingSQLExecutor::new("sqlite", "sqlite_exec", remote_ctx(remote, csv).await);
    let queries = executor.queries();
    let provider = Arc::new(SQLFederationProvider::new(Arc::new(executor)));
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("foo", DataType::Utf8, true),
        Field::new("bar", DataType::Int64, true),
    ]));
    let source = Arc::new(SQLTableSource::new_with_schema(
        provider,
        TableReference::bare(remote.to_string()).into(),
        Arc::clone(&arrow_schema),
    ));

    // *With* a fallback provider that accepts filter pushdown. Without one,
    // `push_down_filter` leaves the predicate as a `Filter` node above the
    // scan, and the analyzer rewrites it on the way past. A provider that
    // takes the filter moves it into `TableScan.filters` instead, where only
    // an explicit rewrite reaches it. Every real SQL table provider does
    // this, so it is the shape that matters.
    let fallback = Arc::new(PushdownProvider {
        schema: Arc::clone(&arrow_schema),
    });
    let ctx = SessionContext::new_with_state(datafusion_federation::default_session_state());
    ctx.register_table(
        local,
        Arc::new(FederatedTableProviderAdaptor::new_with_provider(
            source, fallback,
        )),
    )
    .expect("register renamed table");
    (ctx, queries)
}

/// A table provider that exists only to claim filters, so that DataFusion
/// pushes them into the `TableScan` rather than leaving them above it. It is
/// never scanned: federation replaces it before execution.
#[derive(Debug)]
struct PushdownProvider {
    schema: datafusion::arrow::datatypes::SchemaRef,
}

#[async_trait]
impl datafusion::catalog::TableProvider for PushdownProvider {
    fn schema(&self) -> datafusion::arrow::datatypes::SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> datafusion::logical_expr::TableType {
        datafusion::logical_expr::TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&datafusion::logical_expr::Expr],
    ) -> datafusion::error::Result<Vec<datafusion::logical_expr::TableProviderFilterPushDown>> {
        Ok(vec![
            datafusion::logical_expr::TableProviderFilterPushDown::Exact;
            filters.len()
        ])
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[datafusion::logical_expr::Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        unreachable!("federation replaces this provider before execution")
    }
}
