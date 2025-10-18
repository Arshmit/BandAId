const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const clickBtn = document.getElementById("clickBtn");
const submitBtn = document.getElementById("submitBtn");
const errorMsg = document.getElementById("errorMsg");

let selectedFile = null;

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const validTypes = ["image/jpeg", "image/png", "image/jpg"];
  const maxSize = 5 * 1024 * 1024; // 5MB limit

  if (!validTypes.includes(file.type)) {
    showError(" Please upload a valid image file (JPG or PNG).");
    selectedFile = null;
  } else if (file.size > maxSize) {
    showError("️ File too large! Must be under 5MB.");
    selectedFile = null;
  } else {
    errorMsg.textContent = "";
    selectedFile = file;
    alert(` ${file.name} selected successfully.`);
  }
});

clickBtn.addEventListener("click", () => {
  alert(" Camera feature coming soon!");
});

submitBtn.addEventListener("click", () => {
  if (!selectedFile) {
    showError("Please upload a valid image before submitting.");
    return;
  }
  alert(`Submitting file: ${selectedFile.name}`);
});

function showError(message) {
  errorMsg.textContent = message;
  setTimeout(() => {
    errorMsg.textContent = "";
  }, 4000);
}

