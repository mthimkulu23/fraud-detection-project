const BACKEND_URL = 'http://127.0.0.1:8000'; // Ensure this matches your FastAPI backend URL

// Get DOM elements
const fraudDetectionForm = document.getElementById('fraudDetectionForm');
const predictButton = document.getElementById('predictButton');
const messageContainer = document.getElementById('messageContainer');
const backendStatusSpan = document.getElementById('backendStatus');
const refreshStatusBtn = document.getElementById('refreshStatusBtn');
const totalTransactionsSpan = document.getElementById('totalTransactions');
const fraudulentTransactionsSpan = document.getElementById('fraudulentTransactions');
const recentPredictionsBody = document.getElementById('recentPredictionsBody');
const currencySelect = document.getElementById('currencySelect');

// Input fields
const amountInput = document.getElementById('amount');
const transactionFrequencyInput = document.getElementById('transactionFrequency');
const locationRiskInput = document.getElementById('locationRisk');
const timeOfDayInput = document.getElementById('timeOfDay');
const isInternationalSelect = document.getElementById('isInternational');
const ipCountryMismatchSelect = document.getElementById('ipCountryMismatch');
const failedAuthAttemptsInput = document.getElementById('failedAuthAttempts'); // New: Get the input element

// Global state for dashboard statistics and recent predictions
let totalTransactions = 0;
let fraudulentTransactions = 0;
let allPredictionsHistory = []; // Stores all predictions for chart data
let recentPredictions = []; // Stores up to 5 most recent predictions for table
let currentCurrencySymbol = 'R'; // Default currency symbol

// D3 Map variables
const mapContainer = d3.select("#worldMapContainer");
const mapTooltip = d3.select("#mapTooltip");
let svg, projection, path;
let worldData; // To store loaded TopoJSON data
let fraudulentCountryCounts = {}; // To store fraud counts per country

// Function to update summary statistics
function updateSummaryStatistics() {
    totalTransactionsSpan.textContent = totalTransactions;
    fraudulentTransactionsSpan.textContent = fraudulentTransactions;
}

// Function to update recent predictions table
function updateRecentPredictionsTable() {
    if (recentPredictions.length === 0) {
        // Updated colspan from 9 to 10 to account for the new column
        recentPredictionsBody.innerHTML = '<tr><td colspan="10" class="text-center py-4 text-gray-500">No predictions yet.</td></tr>';
        return;
    }

    recentPredictionsBody.innerHTML = recentPredictions.map(p => {
        const fraudStatusClass = p.is_fraud ? 'text-red-600 font-bold' : 'text-green-600 font-bold';
        return `
            <tr>
                <td>${currentCurrencySymbol}${p.amount.toFixed(2)}</td>
                <td>${p.transaction_frequency_24h}</td>
                <td>${p.location_risk_score.toFixed(1)}</td>
                <td>${p.time_of_day_hour}h</td>
                <td>${p.is_international ? 'Yes' : 'No'}</td>
                <td>${p.ip_country_mismatch ? 'Yes' : 'No'}</td>
                <td>${p.simulated_category || 'N/A'}</td>
                <td>${p.simulated_location_name || 'N/A'}</td>
                <td>${p.failed_auth_attempts}</td> <!-- Display the new field -->
                <td class="${fraudStatusClass}">${p.is_fraud ? 'Fraud' : 'No Fraud'}</td>
            </tr>
        `;
    }).join('');
}

// Function to display messages (errors or results)
function displayMessage(type, message) {
    messageContainer.innerHTML = ''; // Clear previous messages
    const div = document.createElement('div');
    div.className = `mt-6 p-4 rounded-xl ${
        type === 'error' ? 'bg-red-100 border border-red-400 text-red-700' :
        'bg-blue-50 border border-blue-200 text-blue-800'
    }`;
    div.innerHTML = message;
    messageContainer.appendChild(div);
}

// Function to update backend status display
async function updateBackendStatus() {
    backendStatusSpan.className = 'px-4 py-2 rounded-full text-sm font-bold bg-yellow-100 text-yellow-800';
    backendStatusSpan.textContent = 'Checking...';
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            if (data.status === 'healthy') {
                backendStatusSpan.className = 'px-4 py-2 rounded-full text-sm font-bold bg-green-100 text-green-800';
                backendStatusSpan.textContent = 'Healthy';
            } else {
                backendStatusSpan.className = 'px-4 py-2 rounded-full text-sm font-bold bg-red-100 text-red-800';
                backendStatusSpan.textContent = 'Unhealthy';
            }
        } else {
            backendStatusSpan.className = 'px-4 py-2 rounded-full text-sm font-bold bg-red-100 text-red-800';
            backendStatusSpan.textContent = 'Unreachable';
        }
    } catch (error) {
        console.error('Backend health check failed:', error);
        backendStatusSpan.className = 'px-4 py-2 rounded-full text-sm font-bold bg-red-100 text-red-800';
        backendStatusSpan.textContent = 'Error (Check Console)';
    }
}

// --- D3 World Map Functions ---
function drawWorldMap() {
    const width = mapContainer.node().clientWidth;
    const height = mapContainer.node().clientHeight;

    svg = mapContainer.append("svg")
        .attr("width", width)
        .attr("height", height);

    projection = d3.geoMercator()
        .scale(width / (2 * Math.PI))
        .translate([width / 2, height / 1.5]); // Adjust vertical translation for better centering

    path = d3.geoPath().projection(projection);

    d3.json("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json").then(function(world) {
        worldData = world; // Store world data globally
        const countries = topojson.feature(world, world.objects.countries).features;

        svg.selectAll("path")
            .data(countries)
            .enter().append("path")
            .attr("class", "country")
            .attr("d", path)
            .attr("data-name", d => d.properties.name) // Store country name
            .on("mouseover", function(event, d) {
                d3.select(this).classed("hovered", true);
                mapTooltip.html(`<strong>${d.properties.name}</strong><br>Fraud: ${fraudulentCountryCounts[d.properties.name] || 0}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px")
                    .style("opacity", 1);
            })
            .on("mousemove", function(event) {
                mapTooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function() {
                d3.select(this).classed("hovered", false);
                mapTooltip.style("opacity", 0);
            })
            .on("click", function(event, d) {
                // Simulate "specific target fraud" by logging to console
                console.log(`Clicked on ${d.properties.name}. Simulating drill-down for fraud analysis.`);
                displayMessage('success', `Simulating drill-down for <strong>${d.properties.name}</strong>. Check console for details.`);
                // In a real SIEM, this would trigger a detailed report or filter
            });

        updateFraudulentLocationsMap(); // Initial map update
    }).catch(error => {
        console.error("Error loading world map data:", error);
        displayMessage('error', 'Failed to load world map data.');
    });
}

// Debounce utility function
function debounce(func, delay) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), delay);
    };
}

// Debounced version of the map update function
const debouncedUpdateFraudulentLocationsMap = debounce(function() {
    // Reset all countries to default color
    svg.selectAll(".country").classed("fraud-highlight", false);

    // Recalculate fraudulentCountryCounts
    fraudulentCountryCounts = {};
    allPredictionsHistory.forEach(p => {
        if (p.is_fraud && p.simulated_location_name) {
            fraudulentCountryCounts[p.simulated_location_name] = (fraudulentCountryCounts[p.simulated_location_name] || 0) + 1;
        }
    });

    // Highlight countries with fraudulent transactions
    svg.selectAll(".country")
        .filter(d => fraudulentCountryCounts[d.properties.name] > 0)
        .classed("fraud-highlight", true);
}, 200); // 200ms debounce delay

// Function to update all charts and map
function updateAllCharts() {
    debouncedUpdateFraudulentLocationsMap(); // Call the debounced map update function
}

// Event listener for form submission
fraudDetectionForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent default form submission

    predictButton.disabled = true;
    predictButton.textContent = 'Predicting...';
    messageContainer.innerHTML = ''; // Clear previous messages

    // Gather form data
    const transactionData = {
        amount: parseFloat(amountInput.value),
        transaction_frequency_24h: parseInt(transactionFrequencyInput.value),
        location_risk_score: parseFloat(locationRiskInput.value),
        time_of_day_hour: parseInt(timeOfDayInput.value),
        is_international: parseInt(isInternationalSelect.value),
        ip_country_mismatch: parseInt(ipCountryMismatchSelect.value),
        failed_auth_attempts: parseInt(failedAuthAttemptsInput.value), // Include new input field value
    };

    try {
        const response = await fetch(`${BACKEND_URL}/predict_fraud`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const isFraudClass = data.is_fraud ? 'text-red-600' : 'text-green-600';
        const resultHtml = `
            <h2 class="text-xl font-semibold text-blue-800 mb-3">Prediction Result:</h2>
            <p class="text-lg text-gray-800">
                <span class="font-medium">Is Fraud: </span>
                <span class="font-bold ${isFraudClass}">
                    ${data.is_fraud ? 'YES' : 'NO'}
                </span>
            </p>
            <p class="text-lg text-gray-800">
                <span class="font-medium">Fraud Probability: </span>
                <span class="font-bold text-indigo-700">
                    ${(data.fraud_probability * 100).toFixed(2)}%
                </span>
            </p>
            <p class="text-md text-gray-600 mt-2">
                ${data.message}
            </p>
        `;
        displayMessage('success', resultHtml);

        // Update summary statistics and recent predictions
        totalTransactions++;
        if (data.is_fraud) {
            fraudulentTransactions++;
        }
        updateSummaryStatistics();

        // Add to all history and recent predictions
        const fullPredictionData = {
            ...transactionData,
            is_fraud: data.is_fraud,
            simulated_category: getSimulatedCategory(transactionData),
            simulated_location_name: getSimulatedLocationName(transactionData) // Add simulated location
        };
        allPredictionsHistory.push(fullPredictionData);
        recentPredictions.unshift(fullPredictionData);
        if (recentPredictions.length > 5) {
            recentPredictions.pop();
        }
        updateRecentPredictionsTable();
        updateAllCharts(); // Update charts and map after new data

    } catch (error) {
        console.error('Prediction failed:', error);
        displayMessage('error', `<p class="font-bold">Prediction Error:</p><p>${error.message}</p>`);
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = 'Predict Fraud';
    }
});

// Function to assign a simulated category based on transaction data
// This is a simple heuristic for demonstration, not a real ML classification
function getSimulatedCategory(transaction) {
    if (transaction.amount > 10000 && transaction.location_risk_score > 7) {
        return 'ATM Skimming';
    }
    if (transaction.transaction_frequency_24h > 10 && transaction.amount < 100) {
        return 'Card Testing';
    }
    if (transaction.is_international && transaction.time_of_day_hour < 6) {
        return 'Account Takeover';
    }
    if (transaction.amount > 5000 && transaction.ip_country_mismatch) {
        return 'Identity Theft';
    }
    return 'Legitimate Transaction';
}

// Function to assign a simulated location name based on transaction data
// This maps to actual country names found in world-atlas TopoJSON
function getSimulatedLocationName(transaction) {
    if (transaction.is_international) {
        const internationalCountries = ['Nigeria', 'Russia', 'China', 'Brazil', 'India', 'United Kingdom', 'Germany', 'Australia'];
        // Assign a random international country for each international transaction for variety
        return internationalCountries[Math.floor(Math.random() * internationalCountries.length)];
    }
    // Assume local transactions are in South Africa for SA banks context
    return 'South Africa';
}

// Event listener for currency selection
currencySelect.addEventListener('change', (event) => {
    currentCurrencySymbol = event.target.value;
    // Update the amount label immediately
    document.querySelector('label[for="amount"]').textContent = `Amount (${currentCurrencySymbol})`;
    // Re-render the recent predictions table to reflect the new currency symbol
    updateRecentPredictionsTable();
});

// Event listener for refresh status button
refreshStatusBtn.addEventListener('click', updateBackendStatus);

// Initial setup when the page loads
document.addEventListener('DOMContentLoaded', () => {
    updateBackendStatus();
    updateSummaryStatistics(); // Initialize with 0
    updateRecentPredictionsTable(); // Initialize as empty
    drawWorldMap(); // Draw the map on load
    updateAllCharts(); // Initialize charts

    // Set initial currency symbol in the label
    document.querySelector('label[for="amount"]').textContent = `Amount (${currentCurrencySymbol})`;
});
