"use strict";  // enables more strict behavior of JS

// the way of adding the Event listener - to guarantee that the page and all elements are available
document.addEventListener("DOMContentLoaded", function () {
    // console.log("Index Page loaded");

    // Set explicitly default values for indexing scheme and m,n associated orders and save handlers to them
    const nOrderInput = document.getElementById("firstInputOrder"); nOrderInput.value = 0;
    const mOrderInput = document.getElementById("secondInputOrder"); mOrderInput.value = 0;

    // Store handles to the DOM elements as constant variables, since they won't change
    const firstOrderLabel = document.getElementById("firstOrderLabel");  // label with n or i 
    const getIndicesBtn = document.getElementById("getIndicesBtn");
    const ordersLabel = document.getElementById("ordersLabel"); 
    const secondOrderLabel = document.getElementById("secondOrderLabel");
    const secondInputOrder = document.getElementById("secondInputOrder");
    const selector = document.getElementsByName("Index type")[0];  // get the selected HTML element by name
    selector.selectedIndex = 0;  // set the default option to selected HTML element
    const conversionReport = document.getElementById("conversionString");

    // Variables for storing values
    let nOrder = 0; let mOrder = 0; let osaIndex = -1; let nollIndex = -1; let fringeIndex = -1;
    let notOrdersSelected = false; let selectedNollOrFringe = false; let selectedPolynomialType = selector.value;
    let previousOrderN = 0; let previousOrderM = 0;

    // Register event for tracing the new selected value of indexing type for Zernike polynomial from index.html
    selector.addEventListener("change", () => {
        conversionReport.textContent = "Click button on the left to get here conversions";
        console.log("Selected type of polynomial specification: " + selector.value); 
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        nOrder = -1; mOrder = -1; osaIndex = -1; nollIndex = -1; fringeIndex = -1;  // set inconsistent values as defaults
        // Adjust inputs depending on the selected type of polynomial indexing
        switch (selectedPolynomialType){
            case "m,n":
                nOrder = nOrderInput.value; mOrder = mOrderInput.value;
                ordersLabel.innerText = "orders"; firstOrderLabel.innerText = "n =";
                secondOrderLabel.style.visibility = "visible"; secondInputOrder.style.visibility = "visible"; 
                notOrdersSelected = false; selectedNollOrFringe = false; 
                if (nOrder === -1 || mOrder === -1) {
                    nollIndex = 0; mOrder = 0; // Set the appropriate default value
                }
                break;
            case "osa":
                osaIndex = nOrderInput.value; firstOrderLabel.innerText = "j =";
                notOrdersSelected = true; selectedNollOrFringe = false;
                if (osaIndex === -1){
                    osaIndex = 0;  // Set the appropriate default value
                }
                break;
            case "noll":
                nollIndex = nOrderInput.value;
                notOrdersSelected = true; selectedNollOrFringe = true;
                if (nollIndex === -1 || nollIndex === 0) {
                    nollIndex = 1;  // Set the appropriate default value
                }
                break;
            case "fringe":
                fringeIndex = nOrderInput.value;
                notOrdersSelected = true; selectedNollOrFringe = true;
                if (fringeIndex === -1 || fringeIndex === 0) {
                    fringeIndex = 1;  // Set the appropriate default value
                }
                break;
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

    // Register update of nOrder / OSA, Noll, Fringe indices (value is used for both cases: n order and the specific index)
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

    // Register update of mOrder (disabled then any of indices selected)
    mOrderInput.addEventListener("change", () => {
        conversionReport.textContent = "Click button on the left to get here conversions";
        let isNumber = validateOrderN(); 
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        switch (selectedPolynomialType){
            case "m,n":
                if (isNumber && validateOrderM()) {
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
            if (inputValue >= 0 && inputValue <= 200){
                return true;
            } else if (inputValue > 200){
                window.alert("Order or index more than 200 is too high and not allowed!");
            }
        }
        return false;
    }

    // Validate input for mOrderInput
    function validateOrderM(){
        let inputValue = parseInt(mOrderInput.value);
        if (Number.isInteger(inputValue)){
            if (inputValue >= -200 && inputValue <= 200){
                return true;
            } else if (inputValue > 200){
                window.alert("Order or index more than 200 is too high and not allowed!");
            }
        }
        return false;
    }

    // Handle clicking on the button for getting indices and orders
    getIndicesBtn.addEventListener("click", () =>{
        // console.log(`Stored values: n=${nOrder}, m=${mOrder}, OSA=${osaIndex}, Noll=${nollIndex}, Fringe=${fringeIndex}`);
        selectedPolynomialType = selector.value;  // update value each time from the HTML selector
        switch (selectedPolynomialType){
            case "m,n":
                if (checkOrders(mOrder, nOrder)) {
                    console.log("Orders checked successfully");
                    convertOrders();
                    conversionReport.textContent = `OSA index = ${osaIndex}, Noll index = ${nollIndex}, Fringe index = ${fringeIndex}`;
                    previousOrderN = Number(nOrderInput.value); previousOrderM = Number(mOrderInput.value);  // explicit conversion to Number
                } else {
                    conversionReport.textContent = "Orders provided inconsistently and reassigned to the previous checked ones";
                    mOrder = previousOrderM; nOrder = previousOrderN; nOrderInput.value = previousOrderN; mOrderInput.value = previousOrderM; 
                }
                break;
            case "osa":
                index2orders(selectedPolynomialType); 
                conversionReport.textContent = `m = ${mOrder}, n = ${nOrder}, Noll index = ${nollIndex}, Fringe index = ${fringeIndex}`;
                break;
            case "noll":
                index2orders(selectedPolynomialType); 
                conversionReport.textContent = `m = ${mOrder}, n = ${nOrder}, OSA index = ${osaIndex}, Fringe index = ${fringeIndex}`;
                break;
            case "fringe":
                index2orders(selectedPolynomialType); 
                conversionReport.textContent = `m = ${mOrder}, n = ${nOrder}, OSA index = ${osaIndex}, Noll index = ${nollIndex}`;
                break;
        }
    });

    // Check (validate) provided orders - their consistency
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

    // Conversion of radial, angular orders to set of indices as response to click of the button
    function convertOrders(){
        osaIndex = convertOrders2OSA(mOrder, nOrder); 
        nollIndex = convertOrders2Noll(mOrder, nOrder);
        fringeIndex = convertOrders2Fringe(mOrder, nOrder);
    }

    // Conversion orders to OSA / ANSI index
    function convertOrders2OSA(m, n){
        return (n*(n + 2) + m)/2;
    }

    // Conversion orders to Fringe index
    function convertOrders2Fringe(m, n){
        let add_last = 0; 
        if (m < 0){
            add_last = 1;
        }
        return Math.pow((1+(n + Math.abs(m))/2), 2) - 2*Math.abs(m) + add_last;
    }

    // Conversion orders to Noll index
    function convertOrders2Noll(m, n){
        let add_n = 1;
        if (m > 0){
            if (n % 4 === 0){
                add_n = 0;
            }
            else if((n - 1) % 4 === 0){
                add_n = 0;
            }      
        }
        else if (m < 0){
            if ((n - 2) % 4 === 0){
                add_n = 0;
            } 
            else if ((n - 3) % 4 === 0){
                add_n = 0;
            }
        }
        return (n*(n + 1))/2 + Math.abs(m) + add_n;
    }

    // Convert index to orders
    function index2orders(indexType){
        let found = false;
        for(let radialOrder = 0; radialOrder <= 200; radialOrder++){
            let m = -radialOrder; let n = radialOrder; 
            for(let polynomial = 0; polynomial <= radialOrder; polynomial++){
                switch(indexType){
                    case "osa":
                        if (osaIndex === convertOrders2OSA(m, n)){
                            found = true; mOrder = m; nOrder = n;
                            nollIndex = convertOrders2Noll(mOrder, nOrder);
                            fringeIndex = convertOrders2Fringe(mOrder, nOrder);
                        }
                        break;
                    case "noll":
                        if (nollIndex === convertOrders2Noll(m, n)){
                            found = true; mOrder = m; nOrder = n;
                            osaIndex = convertOrders2OSA(mOrder, nOrder);
                            fringeIndex = convertOrders2Fringe(mOrder, nOrder);
                        }
                        break;
                    case "fringe":
                        if (fringeIndex === convertOrders2Fringe(m, n)){
                            found = true; mOrder = m; nOrder = n;
                            osaIndex = convertOrders2OSA(mOrder, nOrder);
                            nollIndex = convertOrders2Noll(mOrder, nOrder);
                        }
                        break;
                }
                if (found) break;
                m += 2; 
            }
            if (found) break;      
        }    
    }

});
