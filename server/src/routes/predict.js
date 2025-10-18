// src/routes/predict.js
import { Router } from "express";
import multer from "multer";
import crypto from "crypto";

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5 MB
});

router.post("/", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "image is required" });
    }

    let result;
    const mlUrl = process.env.ML_SERVICE_URL;

    if (mlUrl) {
      // Forward raw bytes to your ML microservice (FastAPI, Flask, etc.)
      const resp = await fetch(`${mlUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: req.file.buffer,
      });
      result = await resp.json();
    } else {
      // Stubbed scoring for hackathon/dev mode
      const risk_score = Number((Math.random() * 100).toFixed(2));

      // Match frontend expected structure
      const label = risk_score > 70 ? "Infected" : risk_score > 40 ? "At Risk" : "Healthy";
      result = {
        prediction: label,
        confidence: (risk_score / 100).toFixed(2),
        probabilities: {
          Healthy: 1 - risk_score / 100,
          "At Risk": risk_score / 150,
          Infected: risk_score / 200,
        },
      };
    }

    res.json({
      id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
      filename: req.file.originalname,
      size: req.file.size,
      contentType: req.file.mimetype,
      ...result,
    });
  } catch (err) {
    next(err);
  }
});

export default router;

