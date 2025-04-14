// Example video handling functions

// Function to load example videos
function loadExampleVideo(exampleId) {
    // Hide previous results if any
    $("#results-container").addClass("d-none");
    
    // Show loading indicator
    $("#example-loading").removeClass("d-none");
    
    // Make AJAX request to load the example
    $.ajax({
        url: "/api/example",
        type: "GET",
        data: { id: exampleId },
        success: function(response) {
            if (response.status === "success") {
                // Update UI to show example has been loaded
                $("#example-name").text(response.name);
                $("#example-info").text(response.description);
                $("#example-container").removeClass("d-none");
                
                // Clear file input
                $("#videos").val("");
                
                // Enable analyze button and set data attribute
                $("#analyze-btn")
                    .prop("disabled", false)
                    .data("example-id", exampleId);
                
                // Set mode to example
                $("#upload-mode").val("example");
            } else {
                // Show error
                alert("Error loading example: " + response.error);
            }
            
            // Hide loading indicator
            $("#example-loading").addClass("d-none");
        },
        error: function() {
            alert("Error loading example. Please try again.");
            $("#example-loading").addClass("d-none");
        }
    });
}

// Function to analyze example video
function analyzeExampleVideo(exampleId) {
    // Show progress
    $("#progress").removeClass("d-none");
    $("#progress-text").text("Processing example video...");
    $("#results-container").addClass("d-none");
    $("#analyze-btn").prop("disabled", true);
    
    // Make AJAX request
    $.ajax({
        url: "/api/analyze_example",
        type: "POST",
        data: { id: exampleId },
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
}