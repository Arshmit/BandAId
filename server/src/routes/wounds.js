import { Router } from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const router = Router();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_PATH = path.resolve(__dirname, "../../data/wounds.json");

let records = [];
try {
  if (fs.existsSync(DATA_PATH)) {
    records = JSON.parse(fs.readFileSync(DATA_PATH, "utf-8"));
  }
} catch (e) {
  console.warn("Could not load wounds.json:", e.message);
}

router.get("/", (req, res) => {
  const page = Math.max(1, Number(req.query.page) || 1);
  const limit = Math.min(100, Math.max(1, Number(req.query.limit) || 20));
  const q = (req.query.q || "").toString().toLowerCase();
  let rows = records;

  if (q) {
    rows = rows.filter(r =>
      ["id", "image_name", "field", "answer"]
        .map(k => (r[k] || "").toString().toLowerCase())
        .some(v => v.includes(q))
    );
  }

  const start = (page - 1) * limit;
  res.json({ page, limit, total: rows.length, data: rows.slice(start, start + limit) });
});

router.get("/:id", (req, res) => {
  const row = records.find(r => r.id === req.params.id);
  if (!row) return res.status(404).json({ error: "not found" });
  res.json(row);
});

export default router;
