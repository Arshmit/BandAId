// Handles requests to routes that don't exist
export function notFound(req, res, next) {
  res.status(404).json({
    success: false,
    message: `Not Found - ${req.originalUrl}`,
  });
}

// Handles any thrown errors in routes
export function errorHandler(err, req, res, next) {
  console.error("Error:", err.stack || err.message);
  res.status(500).json({
    success: false,
    message: err.message || "Internal Server Error",
  });
}

