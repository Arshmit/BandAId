import { Router } from "express";
import fs from "fs";
import path from "path";
import csv from "csv-parser";

const router = Router();

// Change this to your real folder
const DATA_FOLDER = "/Users/harnoorkaur/Desktop/Final_dataset";

// Utility to load CSV dynamically based on split
async function loadCSV(split = "train") {
  const filename = `final_wound_dataset_${split}.csv`;
  const filePath = path.join(DATA_FOLDER, filename);
  const rows = [];

  if (!fs.existsSync(filePath)) {
    throw new Error(`Dataset file not found: ${filePath}`);
  }

  return new Promise((resolve, reject) => {
    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", row => rows.push(row))
      .on("end", () => resolve(rows))
      .on("error", reject);
  });
}

// GET /api/wounds?split=train&q=abdomen&limit=5
router.get("/", async (req, res, next) => {
  try {
    const split = req.query.split || "train";
    const q = (req.query.q || "").toLowerCase();
    const limit = Number(req.query.limit) || 20;

    const records = await loadCSV(split);
    console.log("Sample record keys:", Object.keys(records[0])); // 👈 ADD THIS HERE

    let filtered = records;
    if (q) {
      filtered = records.filter(r =>
        Object.values(r)
          .join(" ")
          .toLowerCase()
          .includes(q)
      );
    }

    res.json({
      split,
      total: filtered.length,
      data: filtered.slice(0, limit),
    });
  } catch (err) {
    next(err);
  }
});

export default router;

