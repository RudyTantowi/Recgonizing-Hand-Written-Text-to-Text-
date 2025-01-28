document
  .getElementById("upload-form")
  .addEventListener("submit", function (event) {
    const fileInput = document.getElementById("file-input");
    if (!fileInput.value) {
      event.preventDefault();
      alert("Please select a file before uploading!");
    }
  });

document
  .getElementById("file-input")
  .addEventListener("change", function (event) {
    const file = event.target.files[0]; // Ambil file yang diunggah
    const preview = document.getElementById("image-preview"); // Elemen img
    const previewContainer = document.getElementById("image-preview-container");

    if (file) {
      const reader = new FileReader();

      // Ketika file selesai dibaca
      reader.onload = function (e) {
        preview.src = e.target.result; // Set src ke data URL gambar
        preview.style.display = "block"; // Tampilkan gambar
      };

      reader.readAsDataURL(file); // Membaca file
    } else {
      preview.style.display = "none"; // Sembunyikan gambar jika tidak ada file
    }
  });
