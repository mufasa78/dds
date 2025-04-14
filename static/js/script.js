$(document).ready(function() {
    // Check model loading status every 2 seconds
    if ($("#model-loading").length) {
        checkModelStatus();
    }
    
    // Handle form submission
    $("#upload-form").on("submit", function(e) {
        e.preventDefault();
        
        const fileInput = $("#videos")[0];
        if (fileInput.files.length === 0) {
            alert("Please select at least one video file.");
            return;
        }
        
        // Create FormData
        const formData = new FormData();
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append("videos", fileInput.files[i]);
        }
        
        // Show progress
        $("#analyze-btn").prop("disabled", true);
        $("#progress").removeClass("d-none");
        $("#progress-text").text("Processing videos...");
        $("#results-container").addClass("d-none");
        $("#results").empty();
        
        // Send AJAX request
        $.ajax({
            url: "/upload",
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                displayResults(response.results);
            },
            error: function(xhr) {
                let errorMsg = "An error occurred during processing.";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMsg = xhr.responseJSON.error;
                }
                alert(errorMsg);
                $("#progress").addClass("d-none");
                $("#analyze-btn").prop("disabled", false);
            }
        });
    });
    
    // Function to check model loading status
    function checkModelStatus() {
        $.getJSON("/api/model_status", function(data) {
            if (data.loaded) {
                $("#model-loading").fadeOut(function() {
                    $(this).remove();
                });
                $("#analyze-btn").prop("disabled", false);
            } else {
                setTimeout(checkModelStatus, 2000);
            }
        });
    }
    
    // Function to display results
    function displayResults(results) {
        $("#progress").addClass("d-none");
        $("#results-container").removeClass("d-none");
        $("#analyze-btn").prop("disabled", false);
        
        for (const [filename, result] of Object.entries(results)) {
            let statusClass;
            let icon;
            
            if (result.prediction === "Real") {
                statusClass = "real";
                icon = '<i class="bi bi-check-circle-fill text-success me-2"></i>';
            } else if (result.prediction === "Fake") {
                statusClass = "fake";
                icon = '<i class="bi bi-x-circle-fill text-danger me-2"></i>';
            } else {
                statusClass = "error";
                icon = '<i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>';
            }
            
            const confidence = Math.round(result.confidence);
            const resultItem = `
                <div class="result-item ${statusClass}">
                    <div>
                        <h5>${filename}</h5>
                        <p class="mb-1">${icon} ${result.prediction}</p>
                        <div class="d-flex align-items-center">
                            <span class="me-2">Confidence: ${confidence}%</span>
                            <div class="confidence-gauge">
                                <div class="confidence-level" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            $("#results").append(resultItem);
        }
    }
});