import warnings


# Suppress third-party pending deprecation noise from langgraph/langchain internals.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version\..*",
    category=Warning,
)
