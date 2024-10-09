"use strict";

// Perform all actions below when the page is loaded
document.addEventListener("DOMContentLoaded", () => {

    // Selectors of DOM elements
    const images = document.querySelectorAll(".ZernProfile");  // returns the NodeList class
    const nameImage = document.getElementById("infoStringProfiles");
    const imagesContainer = document.getElementById("profilesContainer");
    const pyramidSection = document.getElementById("pyramid-section");
    const pageBody = document.querySelector("body");

    // Variables accessible for all defined below functions, mostly DOM elements
    let defaultNameImageText = nameImage.textContent;  // default text for element showing the polynomial name

    // Make the pyramidal order for the images in the recallable function
    function makePyramid(init_y_shift, x_reduce_distance){
        let i = 0; let y_shift = 20 + init_y_shift;  let x_shift = 80 - x_reduce_distance; let y_step = 120; let element_width = 100;
        let elementsInRow = 2; let half_width_main = Math.round(imagesContainer.clientWidth/2);
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
                    let height = imagesContainer.clientHeight;
                    imagesContainer.style.height = `${height*(n_row-0.05) + y_shift}px`;
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
        
    // Add event listeners to each image for changing the represented on the page element to represent their name
    images.forEach(element => {
        let namePrefix = "Pointer on the image of:  ";
        element.addEventListener("mouseenter", (event) => {
            // switch statement for getting id of image container
            switch(element.id){
                case "ZernOsa1":
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
        });
    });

    // Adaptive placing of images in columns or in a pyramid
    if (visualViewport.width > 1550) {
        makePyramid(0, 0);
        expandBody();  // check if the body element is less than the screen size and force to fill it
    } else if((visualViewport.width < 1550) && (visualViewport.width > 1000)){
        makePyramid(25, 0);
    } else if((visualViewport.width < 1000) && (visualViewport.width > 700)){
        makePyramid(25, 60);
    } else if((visualViewport.width < 700) && (visualViewport.width > 555)){
        makePyramid(25, 76);
    } else if (visualViewport.width <= 555) {
        pyramidSection.style.display = "none"; 
    }

    // Force body to fill the entire screen if there is not enough content even. Taken from "index_page.js" script
    function expandBody() {
        let hW = window.innerHeight; let hBody = pageBody.offsetHeight; 
        if (hW > hBody) {
            pageBody.classList.add("h-screen");
        }
    }
});
