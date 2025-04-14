$(document).ready(function() {
    // Check model loading status every 2 seconds
    if ($("#model-loading").length) {
        checkModelStatus();
    }
    
    // Handle form submission
    $("#upload-form").on("submit", function(e) {
        e.preventDefault();
        
        // Check if we're in example mode
        const mode = $("#upload-mode").val();
        if (mode === "example") {
            // Process example video
            const exampleId = $("#analyze-btn").data("example-id");
            if (exampleId) {
                analyzeExampleVideo(exampleId);
                return;
            }
        }
        
        // Otherwise, process uploaded files
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
            
            // Processing stats
            let statsHtml = '';
            if (result.stats) {
                const stats = result.stats;
                statsHtml = `
                    <div class="mt-3 small text-muted">
                        <div class="row">
                            ${stats.processing_time ? `
                                <div class="col-md-4">
                                    <strong>Processing Time:</strong> ${stats.processing_time.toFixed(2)}s
                                </div>
                            ` : ''}
                            
                            ${stats.frames_sampled ? `
                                <div class="col-md-4">
                                    <strong>Frames Analyzed:</strong> ${stats.frames_sampled}
                                </div>
                            ` : ''}
                            
                            ${stats.sampling_strategy ? `
                                <div class="col-md-4">
                                    <strong>Sampling:</strong> ${stats.sampling_strategy}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
            
            const resultItem = `
                <div class="result-item card mb-3 ${statusClass}">
                    <div class="card-body">
                        <h5 class="card-title">${filename}</h5>
                        <div class="mb-3">
                            <span class="badge ${result.prediction === "Real" ? "bg-success" : (result.prediction === "Fake" ? "bg-danger" : "bg-warning")} fs-6">
                                ${icon} ${result.prediction}
                            </span>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Confidence: ${confidence}%</label>
                            <div class="progress">
                                <div class="progress-bar ${result.prediction === "Real" ? "bg-success" : "bg-danger"}" 
                                     role="progressbar" 
                                     style="width: ${confidence}%" 
                                     aria-valuenow="${confidence}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                    ${confidence}%
                                </div>
                            </div>
                        </div>
                        ${statsHtml}
                    </div>
                </div>
            `;
            
            $("#results").append(resultItem);
        }
    }
});