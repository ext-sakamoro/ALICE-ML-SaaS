//! ALICE ML SaaS — API Gateway
//!
//! リクエストルーティング、レート制限、CORS処理を担当。

use axum::{
    body::Body,
    http::{Request, StatusCode, Uri},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use serde::Serialize;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::EnvFilter;

use std::net::SocketAddr;

#[derive(Serialize)]
struct GatewayHealth {
    status: String,
    gateway_version: String,
    upstream: String,
}

async fn health() -> Json<GatewayHealth> {
    Json(GatewayHealth {
        status: "ok".to_string(),
        gateway_version: env!("CARGO_PKG_VERSION").to_string(),
        upstream: "http://localhost:8081".to_string(),
    })
}

async fn proxy(req: Request<Body>) -> Result<Response, StatusCode> {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let upstream = format!("http://localhost:8081{path}{query}");

    let client = reqwest::Client::new();
    let method = req.method().clone();

    let body_bytes = match axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return Err(StatusCode::BAD_REQUEST),
    };

    let upstream_req = client.request(method, &upstream).body(body_bytes.to_vec());

    match upstream_req.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = resp.bytes().await.unwrap_or_default();
            Ok((status, body).into_response())
        }
        Err(_) => Err(StatusCode::BAD_GATEWAY),
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .fallback(proxy)
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("ALICE ML Gateway listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
