import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import healthRouter from "./routes/health.js";
import predictRouter from "./routes/predict.js";
import woundsRouter from "./routes/wounds.js";
import { notFound, errorHandler } from "./middleware/error.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8080;
const FRONTEND_ORIGIN = (process.env.FRONTEND_ORIGIN || "")
  .split(",")
  .map(s => s.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: FRONTEND_ORIGIN.length ? FRONTEND_ORIGIN : "*",
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    credentials: true
  })
);

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) =>
  res.json({ name: "BandAId API", version: "0.1.0", ok: true })
);
app.use("/api/health", healthRouter);
app.use("/api/predict", predictRouter);
app.use("/api/wounds", woundsRouter);

app.use(notFound);
app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`✅ API listening on http://localhost:${PORT}`);
});
