// the way of adding the Event listener - to guarantee that the page and all elements are available
document.addEventListener("DOMContentLoaded", function () {
    console.log("Page loaded");

    // Set explicitly default values for m, n orders
    let nOrderInput = document.getElementById("firstInputOrder"); nOrderInput.value = 0; 

    // register event for tracing the new selected value from index.html
    let selector = document.getElementsByName("Index type")[0];  // get the selected HTML element by name
    selector.selectedIndex = 0;  // set the default option to selected HTML element
    selector.addEventListener("change", () => {
        console.log(selector.value);
        let selectedPolynomialType = selector.value;
        var notOrdersSelected = true; var selectedNollOrFringe = false;
        switch (selectedPolynomialType){
            case "m,n":
                document.getElementById("ordersLabel").innerText = "orders"; 
                document.getElementById("firstOrderLabel").innerText = "n=";
                document.getElementById("secondOrderLabel").style.visibility = "visible"; 
                document.getElementById("secondInputOrder").style.visibility = "visible"; 
                notOrdersSelected = false; selectedNollOrFringe = false; break;
            case "osa":
                nOrderInput.innerText = "j=";
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
        // The code for cases "noll" and "fringe" only
        if (selectedNollOrFringe){
            nOrderInput.innerText = "i="; nOrderInput.min = "1"; nOrderInput.value = 1;
        }
    });

});