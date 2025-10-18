import { Router } from "express";
const router = Router();

router.get("/", (_req, res) => {
  res.json({
    status: "ok",
    env: process.env.NODE_ENV || "development",
    time: new Date().toISOString(),
    uptime_seconds: process.uptime()
  });
});

export default router;
