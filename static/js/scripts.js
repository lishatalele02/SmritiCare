document.addEventListener("DOMContentLoaded", () => {
    // Handle login form submission
    const loginForm = document.getElementById("loginForm");
    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const email = document.getElementById("email").value;
            const password = document.getElementById("password").value;

            if (email && password) {
                window.location.href = "index.html"; // Redirect after login
            } else {
                alert("Please enter valid credentials.");
            }
        });
    }

    // Handle MRI scan upload form submission
    const uploadForm = document.getElementById("uploadForm");
    if (uploadForm) {
        uploadForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const fileInput = document.getElementById("mriUpload");
            const resultDiv = document.getElementById("result");

            if (fileInput.files.length === 0) {
                resultDiv.textContent = "Please upload a valid MRI scan.";
                resultDiv.style.color = "red";
                return;
            }

            resultDiv.textContent = "Analyzing MRI scan...";
            resultDiv.style.color = "#4CAF50";

            setTimeout(() => {
                resultDiv.textContent = "Prediction: Mild Demented";
                resultDiv.style.color = "#333";
            }, 2000);
        });
    }

    // Highlight the active navigation link
    const links = document.querySelectorAll("nav a");
    links.forEach(link => {
        if (link.href === window.location.href) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }

        if (link.textContent.trim().toLowerCase() === "home") {
            link.addEventListener("click", (e) => {
                e.preventDefault();
                window.location.href = link.href; // Redirect to the Home page
            });
        }
    });
});
