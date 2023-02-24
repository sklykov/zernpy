// the way of adding the Event listener - to guarantee that the page and all elements are available
document.addEventListener("DOMContentLoaded", function () {
    console.log("Index Page loaded");
    "use strict";  // enables more strict behavior of JS

    // Variables for storing values
    let nOrder = 0; let mOrder = 0; let osaIndex = -1; let nollIndex = -1; let fringeIndex = -1;
    let notOrdersSelected = false; let selectedNollOrFringe = false;

    // Set explicitly default values for indexing scheme and m,n associated orders
    let nOrderInput = document.getElementById("firstInputOrder"); nOrderInput.value = 0;
    let mOrderInput = document.getElementById("secondInputOrder"); mOrderInput.value = 0;

    // Store handles to the DOM elements as variables
    let firstOrderLabel = document.getElementById("firstOrderLabel");  // label with n or i 
    let getIndicesBtn = document.getElementById("getIndicesBtn");
    let ordersLabel = document.getElementById("ordersLabel"); 
    let secondOrderLabel = document.getElementById("secondOrderLabel");
    let secondInputOrder = document.getElementById("secondInputOrder");
    let selector = document.getElementsByName("Index type")[0];  // get the selected HTML element by name
    selector.selectedIndex = 0;  // set the default option to selected HTML element
    let selectedPolynomialType = selector.value;
    let conversionReport = document.getElementById("conversionString");

    // Register event for tracing the new selected value from index.html
    selector.addEventListener("change", () => {
        conversionReport.textContent = "Click button on the left to get here conversions";
        console.log("Selected type of polynomial specification:" + selector.value); 
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        nOrder = -1; mOrder = -1; osaIndex = -1; nollIndex = -1; fringeIndex = -1;
        // Defines which type of polynomial definition selected
        switch (selectedPolynomialType){
            case "m,n":
                nOrder = nOrderInput.value; mOrder = mOrderInput.value;
                ordersLabel.innerText = "orders"; firstOrderLabel.innerText = "n =";
                secondOrderLabel.style.visibility = "visible"; secondInputOrder.style.visibility = "visible"; 
                notOrdersSelected = false; selectedNollOrFringe = false; break;
            case "osa":
                osaIndex = nOrderInput.value; firstOrderLabel.innerText = "j =";
                notOrdersSelected = true; selectedNollOrFringe = false; break;
            case "noll":
                nollIndex = nOrderInput.value;
                notOrdersSelected = true; selectedNollOrFringe = true; break;
            case "fringe":
                fringeIndex = nOrderInput.value;
                notOrdersSelected = true; selectedNollOrFringe = true; break;
        }
         // The code below is executed if the orders m,n not selected and common for all cases then selected OSA / Noll / Fringe
        if (notOrdersSelected){
            ordersLabel.innerText = "index"; secondOrderLabel.style.visibility = "hidden"; secondInputOrder.style.visibility = "hidden"; 
        }
        // The code for selected cases "noll" or "fringe", except "m,n"
        if (selectedNollOrFringe){
            firstOrderLabel.innerText = "i ="; nOrderInput.min = "1"; nOrderInput.value = 1;
        } else {  // if orders selected
            nOrderInput.min = "0"; nOrderInput.value = 0;
        }
    });

    // Register update of nOrder / indices
    nOrderInput.addEventListener("change", () =>{
        conversionReport.textContent = "Click button on the left to get here conversions";
        let isNumber = validateOrderN(); 
        // console.log("Validation result: " + isNumber);
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        switch (selectedPolynomialType){
            case "m,n":
                if (isNumber) {
                    nOrder = Number(nOrderInput.value); mOrder = Number(mOrderInput.value);
                } else {
                    nOrderInput.value = nOrder; 
                }
                break;
            case "osa":
                if (isNumber) {
                    osaIndex = Number(nOrderInput.value);
                } else {
                    nOrderInput.value = osaIndex; 
                }
                break;
            case "noll":
                if (isNumber) {
                    nollIndex = Number(nOrderInput.value);
                } else {
                    nOrderInput.value = nollIndex; 
                }
                break;
            case "fringe":
                if (isNumber) {
                    fringeIndex = Number(nOrderInput.value);
                } else {
                    nOrderInput.value = fringeIndex; 
                }
                break;
        }
    });

    // Register update of mOrder
    mOrderInput.addEventListener("change", () => {
        conversionReport.textContent = "Click button on the left to get here conversions";
        let isNumber = validateOrderN(); 
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        switch (selectedPolynomialType){
            case "m,n":
                if (isNumber) {
                    nOrder = Number(nOrderInput.value); mOrder = Number(mOrderInput.value);
                } else {
                    mOrderInput.value = mOrder; 
                }
                break;
            default:
                empty;
        }
    });

    // Validate input for nOrderInput
    function validateOrderN(){
        let inputValue = parseInt(nOrderInput.value);
        if (Number.isInteger(inputValue)){
            if (inputValue >= 0 && inputValue <= 100){
                return true;
            }
        }
        return false;
    }

    // Validate input for mOrderInput
    function validateOrderM(){
        let inputValue = parseInt(mOrderInput.value);
        if (Number.isInteger(inputValue)){
            if (inputValue >= 0 && inputValue <= 100){
                return true;
            }
        }
        return false;
    }

    // Handle clicking on the button for getting indices and orders
    getIndicesBtn.addEventListener("click", () =>{
        console.log(`Stored values: n=${nOrder}, m=${mOrder}, OSA=${osaIndex}, Noll=${nollIndex}, Fringe=${fringeIndex}`);
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        switch (selectedPolynomialType){
            case "m,n":
                if (checkOrders(mOrder, nOrder)) {
                    console.log("Orders checked successfully");
                } else {
                    conversionReport.textContent = "Orders provided inconsistently and reassigned to default values";
                    mOrder = 0; nOrder = 0; nOrderInput.value = 0; mOrderInput.value = 0; 
                }
                break;
            case "osa":
                break;
            case "noll":
                break;
            case "fringe":
                break;
        }
    });

    // Check provided orders - their consistency
    function checkOrders(m, n){
        let passedCheck = false;
        // all requirements for orders below
        if (n >= 0){
            if (n > 0){
                if ((n - Math.abs(m)) % 2 == 0){
                    passedCheck = true;
                }
            } else if (m == 0){
                passedCheck = true;
            }
            
        }
        return passedCheck;
    }
});