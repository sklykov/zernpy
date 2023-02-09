// the way of adding the Event listener - to guarantee that the page and all elements are available
document.addEventListener("DOMContentLoaded", function () {
    console.log("Page loaded");

    // Set explicitly default values for indexing scheme and m,n associated orders
    let nOrderInput = document.getElementById("firstInputOrder"); nOrderInput.value = 0;
    let mOrderInput = document.getElementById("secondInputOrder"); mOrderInput.value = 0;
    // Store handles to the DOM elements as variables
    let firstOrderLabel = document.getElementById("firstOrderLabel");  // label with n or i 

    // register event for tracing the new selected value from index.html
    let selector = document.getElementsByName("Index type")[0];  // get the selected HTML element by name
    selector.selectedIndex = 0;  // set the default option to selected HTML element
    selector.addEventListener("change", () => {
        console.log(selector.value);
        let selectedPolynomialType = selector.value;
        let notOrdersSelected = true; let selectedNollOrFringe = false;
        
        switch (selectedPolynomialType){
            case "m,n":
                document.getElementById("ordersLabel").innerText = "orders"; 
                firstOrderLabel.innerText = "n =";
                document.getElementById("secondOrderLabel").style.visibility = "visible"; 
                document.getElementById("secondInputOrder").style.visibility = "visible"; 
                notOrdersSelected = false; selectedNollOrFringe = false; break;
            case "osa":
                firstOrderLabel.innerText = "j =";
                notOrdersSelected = true; selectedNollOrFringe = false; break;
            case "noll":
                notOrdersSelected = true; selectedNollOrFringe = true; break;
            case "fringe":
                notOrdersSelected = true; selectedNollOrFringe = true; break;
        }
         // The code below is executed if the orders m,n not selected and common for all cases then selected OSA / Noll / Fringe
        if (notOrdersSelected){
            document.getElementById("ordersLabel").innerText = "index"; 
            document.getElementById("secondOrderLabel").style.visibility = "hidden"; 
            document.getElementById("secondInputOrder").style.visibility = "hidden"; 
        }
        // The code for selected cases "noll" or "fringe", except "m,n"
        if (selectedNollOrFringe){
            firstOrderLabel.innerText = "i ="; nOrderInput.min = "1"; nOrderInput.value = 1;
        } else {  // if orders selected
            nOrderInput.min = "0"; nOrderInput.value = 0;
        }
    });

});