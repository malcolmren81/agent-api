"""
Prometheus metrics collection.
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from config import settings

# Create a custom registry
registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    "agent_requests_total",
    "Total number of agent requests",
    ["agent_type", "status"],
    registry=registry,
)

request_duration = Histogram(
    "agent_request_duration_seconds",
    "Request duration in seconds",
    ["agent_type"],
    registry=registry,
)

# Model usage metrics
model_selection_count = Counter(
    "model_selection_total",
    "Total model selections",
    ["model_type", "model_name"],
    registry=registry,
)

model_cost = Counter(
    "model_cost_total_usd",
    "Total cost in USD",
    ["model_type", "model_name"],
    registry=registry,
)

# Image generation metrics
image_generation_count = Counter(
    "image_generation_total",
    "Total images generated",
    ["model", "status"],
    registry=registry,
)

image_generation_duration = Histogram(
    "image_generation_duration_seconds",
    "Image generation duration",
    ["model"],
    registry=registry,
)

# Evaluation metrics
evaluation_count = Counter(
    "evaluation_total",
    "Total evaluations performed",
    ["result"],
    registry=registry,
)

evaluation_score = Histogram(
    "evaluation_score",
    "Evaluation scores",
    ["criteria"],
    registry=registry,
)

# Product generation metrics
product_generation_count = Counter(
    "product_generation_total",
    "Total products generated",
    ["product_type", "status"],
    registry=registry,
)

# System metrics
active_tasks = Gauge(
    "active_tasks",
    "Number of currently active tasks",
    registry=registry,
)

# Error metrics
error_count = Counter(
    "errors_total",
    "Total errors",
    ["error_type", "component"],
    registry=registry,
)

# A2A metrics
a2a_requests = Counter(
    "a2a_requests_total",
    "Total A2A requests",
    ["service", "status"],
    registry=registry,
)

a2a_duration = Histogram(
    "a2a_request_duration_seconds",
    "A2A request duration",
    ["service"],
    registry=registry,
)
