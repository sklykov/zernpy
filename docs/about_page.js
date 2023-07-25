"use strict";

// Perform all actions below when the page is loaded
document.addEventListener("DOMContentLoaded", () =>{

    // Selectors of DOM elements
    const images = document.querySelectorAll(".ZernProfile");  // returns the NodeList class
    const navbar = document.getElementsByClassName("navbar")[0];
    const navbarLinks = document.getElementsByClassName("navbarLink");
    const nameImage = document.querySelector(".DisplayZernikeName");

    // Variables accessible for all defined below functions, mostly DOM elements
    let previousPageWidth = false;
    let defaultNameImageText = nameImage.textContent;  // default text for element showing the polynomial name

    // Move the element showing the name of image to the left
    function moveNamePolEl(){
        nameImage.style.textAlign = "left"; 
        nameImage.style.position = "relative";  // allows moving this field below
        nameImage.style.left = "1em"; nameImage.style.top = "1.5em";
        defaultNameImageText = defaultNameImageText.replace("below", "right");
        nameImage.textContent = defaultNameImageText; 
    }

    // Make the pyramidal order for the images in the recallable function
    function makePyramid(init_y_shift, x_reduce_distance){
        let i = 0; let y_shift = 20 + init_y_shift;  let x_shift = 80 - x_reduce_distance; let y_step = 120; let element_width = 100;
        let elementsInRow = 2; let half_width_main = Math.round(document.querySelector(".imagesContainer").clientWidth/2);
        let n_row = 0; let max_row = 2; let initial_x_shift = half_width_main - x_shift - element_width/2 - x_reduce_distance/2;
        // Below - loop over all images queried by querySelectorAll
        images.forEach(element => {
            // element.style.position = "absolute";  // This is useless because it breaks the relative positioning of elements
            // console.log(i);
            element.style.top = `${y_shift}px`; 
            // Below - not that each further row shifted more than previous one
            element.style.left = `${initial_x_shift + i*(x_shift + element_width)}px`;
            // console.log(element.style.left, element.style.top);
            i += 1;
            // move to the next row
            if(i == elementsInRow){
                y_shift += y_step;
                // Below - the adding height to imagesContainer element
                if (n_row < max_row){
                    let height = document.querySelector(".imagesContainer").clientHeight;
                    document.querySelector(".imagesContainer").style.height = `${height*(n_row-0.05) + y_shift}px`;
                }
                n_row += 1;
                // console.log(document.querySelector(".imagesContainer").clientHeight))
                // This makes positioning of the elements in a pyramid + CSS property: position: absolute
                if(n_row % 2 != 0){
                    initial_x_shift -= (n_row*(element_width + x_shift))/2;
                } else {
                    initial_x_shift -= ((n_row-1)*(element_width + x_shift))/2;
                }
                // return values for the next row
                elementsInRow += 1; i = 0;
            }
        });
    }

    // Put all images in columns close to each other for saving space on small monitors
    function putImagesInCols(initial_set){
        let i = 0; let y_shift = 50; let x_shift = 5; let y_step = 110; let element_width = 100;
        let elementsInRow = 3; let half_width_main = Math.round(document.querySelector(".imagesContainer").clientWidth/2);
        let n_row = 0; let max_row = 3; let initial_x_shift = half_width_main - 1.5*(x_shift + element_width);
        // Below - loop over all images queried by querySelectorAll
        images.forEach(element => {
            element.style.top = `${y_shift}px`; 
            element.style.left = `${initial_x_shift + i*(2*x_shift + element_width)}px`;
            // console.log(element.style.left, element.style.top);
            i += 1;
            // move to the next row
            if(i == elementsInRow){
                y_shift += y_step;
                n_row += 1; i = 0;
            }
        });
        // Below - the adding height to imagesContainer element
        if(initial_set){
            let height = document.querySelector(".imagesContainer").clientHeight;
            if(previousPageWidth > 500){
                document.querySelector(".imagesContainer").style.height = `${height + (max_row-3.5)*(element_width)}px`;
            } else {
                document.querySelector(".imagesContainer").style.height = `${height + (max_row+0.5)*(element_width)}px`;
            }
        }
    }
        
    // Add event listeners to each image for changing the represented on the page element to represent their name
    images.forEach(element => {
        let namePrefix = "Pointer on the image of:  ";
        element.addEventListener("mouseenter", (event) => {
            // console.log("Mouse event listener registered");
            // switch statement for getting id of image container
            switch(element.id){
                case "ZernOsa1":
                    // console.log("Should change the naming string");
                    nameImage.textContent = namePrefix + "Vertical Tilt";
                    break;
                case "ZernOsa2":
                    nameImage.textContent = namePrefix + "Horizontal Tilt"; break;
                case "ZernOsa3":
                    nameImage.textContent = namePrefix + "Oblique Astigmatism"; break;
                case "ZernOsa4":
                    nameImage.textContent = namePrefix + "Defocus"; break;
                case "ZernOsa5":
                    nameImage.textContent = namePrefix + "Vertical Astigmatism"; break;
                case "ZernOsa6":
                    nameImage.textContent = namePrefix + "Vertical Trefoil"; break;
                case "ZernOsa7":
                    nameImage.textContent = namePrefix + "Vertical Coma"; break;
                case "ZernOsa8":
                    nameImage.textContent = namePrefix + "Horizontal Coma"; break;
                case "ZernOsa9":
                    nameImage.textContent = namePrefix + "Oblique Trefoil"; break;
            }
        });
        // assign default text to the div element if cursor left the image area
        element.addEventListener("mouseleave", (event) => {
            nameImage.textContent = defaultNameImageText;
            // console.log("Mouseover event");
        });
    });

    // Making the page more responsive - adding the monitoring of size changed (actually, for testing on the browser resizing feature)
    visualViewport.onresize = (event) => {
        // console.log("Event 'resize page' encountered and handled"); 
        if(visualViewport.width < 1550){
            nameImage.style.textAlign = "center"; nameImage.style.position = "static";
            document.querySelector(".flexboxContainer").style.flexDirection = "row";
            defaultNameImageText = defaultNameImageText.replace("right", "below");
            nameImage.textContent = defaultNameImageText; 
            restoreMarginNavbar();
        }
        else{
            moveNamePolEl();  // move the displaying of name string (element) back
            makePyramid(0, 0); // call the put images in pyramid
            document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        if((visualViewport.width < 1550) && (visualViewport.width > 1000)){
            makePyramid(25, 0); // call the put images in pyramid
            document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        if((visualViewport.width < 1000) && (visualViewport.width > 700)){
            makePyramid(25, 60); document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        if((visualViewport.width < 700) && (visualViewport.width > 500)){
            document.querySelector(".flexboxContainer").style.flexDirection = "column";  // change representation of webpage - moving navbar on top
            makePyramid(25, 76); 
            makeLessMarginNavbar();
        }
        if(visualViewport.width <= 500){
            document.querySelector("#RepresentingZernikeStr").textContent = "Representing a few Zernike profiles in 2 columns";
            document.querySelector(".flexboxContainer").style.flexDirection = "column";
            if(previousPageWidth <= 500){
                putImagesInCols(false);
            } else {
                putImagesInCols(true);
            }
            makeLessMarginNavbar();
        }
        previousPageWidth = visualViewport.width; 
    }

    // Specify less vertical margins if navigation bar goes on the top of the page
    function makeLessMarginNavbar(){
        navbar.style.display = "block";  // from flexbox representation to block (takes less space)
        // Somehow, the call navbarLinks.forEach() doesn't work on HTMLCollection
        for (let i=0; i<navbarLinks.length; i++){
            navbarLinks[i].style.paddingTop = "0.25em"; navbarLinks[i].style.paddingBottom = "0.25em"; navbarLinks[i].style.margin = "0.15em";
        }
    }

    // Return back margins (almost the same, better length measures)
    function restoreMarginNavbar(){
        // restore default value for navbar element
        navbar.style.display = "flex"; navbar.style.flexDirection = "column"; 
        // restore default values for navbar elements (links) 
        for (let i=0; i<navbarLinks.length; i++){
            navbarLinks[i].style.paddingTop = "2.5em"; navbarLinks[i].style.paddingBottom = "2.5em"; navbarLinks[i].style.margin = "1.2em 0.5em";
        }
    }

    // Call functions when the page is loaded for initial placing the elements - repeating the logic from above
    if(visualViewport.width > 1550){
        moveNamePolEl(); makePyramid(0, 0);
    } else if((visualViewport.width < 1550) && (visualViewport.width > 1000)){
        makePyramid(25, 0);
    } else if((visualViewport.width < 1000) && (visualViewport.width > 700)){
        makePyramid(25, 60);
    } else if((visualViewport.width < 700) && (visualViewport.width > 500)){
        makePyramid(25, 76); document.querySelector(".flexboxContainer").style.flexDirection = "column";
        makeLessMarginNavbar();
    } else if(visualViewport.width <= 500){
        document.querySelector("#RepresentingZernikeStr").textContent = "Representing a few Zernike profiles in 2 columns";
        // document.querySelector(".flexboxContainer").style.flexDirection = "column";  // shifted to the style media query property
        putImagesInCols(true); makeLessMarginNavbar();
    }
});
