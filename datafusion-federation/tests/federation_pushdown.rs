//! End-to-end federation tests.
//!
//! These mirror what the `df-csv` / `df-csv-advanced` examples demonstrate, but
//! turn it into assertions: each "remote database" is a local DataFusion context
//! that records the SQL it receives, so we can verify both that results are
//! correct and that compute (filters, limits, aggregates, joins) was actually
//! pushed down to the remote rather than run locally.

#![cfg(feature = "sql")]

mod support;

use support::{
    recorded_sql, remote_ctx, row_count, run_federated, schema_provider,
    try_run_without_federation, RecordingSQLExecutor,
};

/// `SELECT *` returns every row of the remote table.
#[tokio::test]
async fn select_star_returns_all_rows() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();
    let schema = schema_provider(std::sync::Arc::new(executor), &["test"]).await;

    let batches = run_federated(schema, "SELECT * FROM test").await;

    assert_eq!(row_count(&batches), 3, "test.csv has 3 rows");
    // The scan was federated: the remote received a SELECT.
    assert!(
        recorded_sql(&queries).contains("select"),
        "remote should have received a query, got: {:?}",
        queries.lock().unwrap()
    );
}

/// A `WHERE` clause is pushed down to the remote as SQL, not applied locally.
#[tokio::test]
async fn filter_is_pushed_down() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();
    let schema = schema_provider(std::sync::Arc::new(executor), &["test"]).await;

    let batches = run_federated(schema, "SELECT * FROM test WHERE bar > 1").await;

    assert_eq!(row_count(&batches), 2, "rows with bar > 1: b,2 and c,3");
    let sql = recorded_sql(&queries);
    assert!(sql.contains("where"), "filter should be pushed down: {sql}");
    assert!(sql.contains("bar"), "predicate column should appear: {sql}");
}

/// A `LIMIT` is pushed down to the remote.
#[tokio::test]
async fn limit_is_pushed_down() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();
    let schema = schema_provider(std::sync::Arc::new(executor), &["test"]).await;

    let batches = run_federated(schema, "SELECT * FROM test LIMIT 1").await;

    assert_eq!(row_count(&batches), 1);
    assert!(
        recorded_sql(&queries).contains("limit"),
        "limit should be pushed down: {:?}",
        queries.lock().unwrap()
    );
}

/// An aggregation is pushed down to the remote.
#[tokio::test]
async fn aggregate_is_pushed_down() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();
    let schema = schema_provider(std::sync::Arc::new(executor), &["test"]).await;

    let batches = run_federated(schema, "SELECT count(*) FROM test").await;

    assert_eq!(row_count(&batches), 1);
    assert!(
        recorded_sql(&queries).contains("count"),
        "aggregate should be pushed down: {:?}",
        queries.lock().unwrap()
    );
}

/// Negative control: the same query WITHOUT the federation rule fails to scan
/// and never reaches the remote. This is the opposite of `filter_is_pushed_down`
/// and proves that the remote is only queried because federation is active.
#[tokio::test]
async fn without_federation_scan_fails_and_remote_is_never_called() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();
    let schema = schema_provider(std::sync::Arc::new(executor), &["test"]).await;

    let result = try_run_without_federation(schema, "SELECT * FROM test WHERE bar > 1").await;

    let err = result.expect_err("scan must fail without the federation rule");
    assert!(
        err.to_string().contains("cannot scan"),
        "expected FederatedTableProviderAdaptor scan error, got: {err}"
    );
    assert!(
        queries.lock().unwrap().is_empty(),
        "remote must not be queried without federation, got: {:?}",
        queries.lock().unwrap()
    );
}

/// A join across two independent remotes federates each side to its own remote,
/// mirroring the cross-database join in `df-csv-advanced`.
#[tokio::test]
async fn cross_provider_join() {
    use datafusion::execution::context::SessionContext;
    use datafusion_federation::sql::MultiSchemaProvider;
    use std::sync::Arc;

    // Remote #1: "sqlite" with table `test_sqlite`.
    let sqlite_ctx = remote_ctx("test_sqlite", "test.csv").await;
    let sqlite_exec = RecordingSQLExecutor::new("sqlite", "sqlite_exec", sqlite_ctx);
    let sqlite_queries = sqlite_exec.queries();
    let sqlite_schema = schema_provider(Arc::new(sqlite_exec), &["test_sqlite"]).await;

    // Remote #2: "postgres" with table `test_pg`.
    let pg_ctx = remote_ctx("test_pg", "test2.csv").await;
    let pg_exec = RecordingSQLExecutor::new("postgres", "postgres_exec", pg_ctx);
    let pg_queries = pg_exec.queries();
    let pg_schema = schema_provider(Arc::new(pg_exec), &["test_pg"]).await;

    let state = datafusion_federation::default_session_state();
    support::overwrite_default_schema(
        &state,
        Arc::new(MultiSchemaProvider::new(vec![sqlite_schema, pg_schema])),
    );
    let ctx = SessionContext::new_with_state(state);

    let batches = support::collect(
        ctx.sql("SELECT t.* FROM test_pg AS t JOIN test_sqlite AS a ON t.foo = a.foo")
            .await
            .expect("plan join"),
    )
    .await;

    // foo in {a,b,c} on both sides -> 3 matching rows.
    assert_eq!(row_count(&batches), 3);

    // Each remote received its own scan; neither remote saw the other's table.
    let sqlite_sql = recorded_sql(&sqlite_queries);
    let pg_sql = recorded_sql(&pg_queries);
    assert!(
        sqlite_sql.contains("test_sqlite"),
        "sqlite remote should scan its table: {sqlite_sql}"
    );
    assert!(
        pg_sql.contains("test_pg"),
        "postgres remote should scan its table: {pg_sql}"
    );
    assert!(
        !sqlite_sql.contains("test_pg"),
        "sqlite remote must not see the postgres table: {sqlite_sql}"
    );
}

/// A join across two engines cannot be folded into one remote query, so
/// each side is federated on its own — and each side should still ask
/// only for the columns the query needs.
///
/// It does not. The federation rule is inserted immediately after
/// `scalar_subquery_to_join`, while `OptimizeProjections` is the *last*
/// rule DataFusion runs, so when a subtree is federated its `TableScan`
/// has not yet been narrowed. A single-table query hides this: the whole
/// plan including its `Projection` is federated, and unparsing that
/// emits the column list. Only when the parent cannot be federated —
/// this case — is the bare scan left to unparse as every column.
///
/// The cost is proportional to table width. Against a real estate this
/// was measured at 40x: 1,048 bytes for two columns fetched directly,
/// 41 KB for the same two columns fetched through a cross-engine join of
/// a 43-column table.
#[tokio::test]
async fn projection_is_pushed_down_across_a_cross_engine_join() {
    let (ctx, alpha_sql, beta_sql) = support::two_remotes().await;

    let batches = support::collect(
        ctx.sql("SELECT a.foo FROM alpha.test a JOIN beta.test2 b ON b.foo = a.foo")
            .await
            .expect("plan query"),
    )
    .await;
    assert_eq!(row_count(&batches), 3, "both sides have the same 3 keys");

    // Only `foo` is needed on either side; `bar` is never referenced.
    let alpha = recorded_sql(&alpha_sql);
    let beta = recorded_sql(&beta_sql);
    assert!(
        !alpha.contains("bar"),
        "alpha fetched a column the query never uses: {alpha}"
    );
    assert!(
        !beta.contains("bar"),
        "beta fetched a column the query never uses: {beta}"
    );
}

/// The same join within *one* engine is folded into a single remote
/// query, and that query is already narrow — because the `Projection`
/// travels with it. This is the contrast that locates the bug: the
/// problem is not joins, it is subtrees whose parent cannot be
/// federated.
#[tokio::test]
async fn projection_is_already_pushed_within_one_engine() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();
    let schema = schema_provider(std::sync::Arc::new(executor), &["test"]).await;

    run_federated(
        schema,
        "SELECT a.foo FROM test a JOIN test b ON b.foo = a.foo",
    )
    .await;

    let sql = recorded_sql(&queries);
    assert!(sql.contains("join"), "the join was federated whole: {sql}");
    assert!(!sql.contains("bar"), "and it is already narrow: {sql}");
}

/// A table registered under a name the remote does not know must have *every*
/// reference rewritten — the `FROM` clause and the filters alike.
///
/// The scan carries its pushed-down filters as a separate field, qualified by
/// the local table name. Rewriting only `table_name` produces SQL whose `FROM`
/// says `test` and whose `WHERE` says `t1`, which PostgreSQL rejects with
/// "missing FROM-clause entry for table t1".
#[tokio::test]
async fn a_renamed_table_rewrites_its_filters_too() {
    let (ctx, queries) = support::renamed_remote("t1", "test", "test.csv").await;

    let batches = support::collect(
        ctx.sql("SELECT foo FROM t1 WHERE bar > 1")
            .await
            .expect("plan query"),
    )
    .await;

    assert_eq!(support::row_count(&batches), 2);
    let sql = support::recorded_sql(&queries);
    assert!(
        !sql.contains("t1"),
        "the local name leaked into the remote SQL: {sql}"
    );
    assert!(
        sql.contains("from test") && sql.contains("test.bar"),
        "filter should be pushed down against the remote name: {sql}"
    );
}

/// The same, under `EXPLAIN ANALYZE`, which executes the plan rather than only
/// planning it — so a filter left pointing at the local name surfaces as a
/// remote error instead of a bad plan nobody ran.
#[tokio::test]
async fn a_renamed_table_can_be_explained_and_analyzed() {
    let (ctx, queries) = support::renamed_remote("t1", "test", "test.csv").await;

    support::collect(
        ctx.sql("EXPLAIN ANALYZE SELECT foo FROM t1 WHERE bar > 1")
            .await
            .expect("plan query"),
    )
    .await;

    let sql = support::recorded_sql(&queries);
    assert!(!sql.is_empty(), "the remote was never asked anything");
    assert!(!sql.contains("t1"), "the local name leaked: {sql}");
}
