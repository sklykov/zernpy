// Perform all actions below when the page is loaded
document.addEventListener("DOMContentLoaded", () =>{
    var previousPageWidth = false;
    var nameImage = document.querySelector(".DisplayZernikeName");
    var defaultNameImageText = nameImage.textContent;

    // Move the element showing the name of image to the left
    function moveNamePolEl(){
        nameImage.style.textAlign = "left"; 
        nameImage.style.position = "relative";  // allows moving this field below
        nameImage.style.left = "15px"; nameImage.style.top = "35px";
    }

    // var images = document.getElementsByClassName("ZernProfile");  // returns some HTMLCollection class, how to loop over the elements?
    // Below - making the pyramidal order for the images in the recallable function
    function makePyramid(init_y_shift, x_reduce_distance){
        var i = 0; var y_shift = 20 + init_y_shift;  var x_shift = 80 - x_reduce_distance; var y_step = 120; var element_width = 100;
        var elementsInRow = 2;
        var half_width_main = Math.round(document.querySelector(".imagesContainer").clientWidth/2);
        var n_row = 0; var max_row = 2; var initial_x_shift = half_width_main - x_shift - element_width/2 - x_reduce_distance/2;
        var images = document.querySelectorAll(".ZernProfile");  // returns the NodeList class
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
                    height = document.querySelector(".imagesContainer").clientHeight;
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

    function putImagesInCols(initial_set){
        var i = 0; var y_shift = 50;  var x_shift = 5; var y_step = 110; var element_width = 100;
        var elementsInRow = 3;
        var half_width_main = Math.round(document.querySelector(".imagesContainer").clientWidth/2);
        var n_row = 0; var max_row = 3; var initial_x_shift = half_width_main - 1.5*(x_shift + element_width);
        var images = document.querySelectorAll(".ZernProfile");  // returns the NodeList class
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
            height = document.querySelector(".imagesContainer").clientHeight;
            if(previousPageWidth > 500){
                document.querySelector(".imagesContainer").style.height = `${height + (max_row-3.5)*(element_width)}px`;
            } else {
                document.querySelector(".imagesContainer").style.height = `${height + (max_row+0.5)*(element_width)}px`;
            }
        }
    }
        
    // Below - add event listeners to each image for changing the string to represent their name
    var images = document.querySelectorAll(".ZernProfile");  // returns the NodeList class
    images.forEach(element => {
        var namePrefix = "Pointer on the image of:  ";
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

    // making the page more responsive - adding the monitoring of size changed
    visualViewport.onresize = (event) => {
        if(visualViewport.width < 1550){
            // below - attempt to input line break, but seems that HTML div element ignores it
            // defaultNameImageText = "Hover mouse over the images (right) \n for getting polynomial name"; 
            // nameImage.textContent = defaultNameImageText;
            nameImage.style.textAlign = "center"; nameImage.style.position = "static";
            document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        else{
            moveNamePolEl();  // move the displaying of name string (element) back
            makePyramid(shift_y=0, x_reduce_distance = 0); // call the put images in pyramid
            document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        if((visualViewport.width < 1550) && (visualViewport.width > 1000)){
            makePyramid(shift_y = 25, x_reduce_distance = 0); // call the put images in pyramid
            document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        if((visualViewport.width < 1000) && (visualViewport.width > 700)){
            makePyramid(shift_y = 25, x_reduce_distance = 60); document.querySelector(".flexboxContainer").style.flexDirection = "row";
            restoreMarginNavbar();
        }
        if((visualViewport.width < 700) && (visualViewport.width > 500)){
            document.querySelector(".flexboxContainer").style.flexDirection = "column";  // change representation of webpage - moving navbar on top
            makePyramid(shift_y = 25, x_reduce_distance = 76); 
            makeLessMarginNavbar();
        }
        if(visualViewport.width <= 500){
            document.querySelector("#RepresentingZernikeStr").textContent = "Representing a few Zernike profiles in 2 columns";
            document.querySelector(".flexboxContainer").style.flexDirection = "column";
            if(previousPageWidth <= 500){
                putImagesInCols(initial_set=false);
            } else {
                putImagesInCols(initial_set=true);
            }
            makeLessMarginNavbar();
        }
        previousPageWidth = visualViewport.width; 
    }

    // Specify less vertical margins if navigation bar goes on the top of the page
    function makeLessMarginNavbar(){
        let navbarLinks = document.getElementsByClassName("navbarLink");
            // Somehow, the call navbarLinks.forEach() doesn't work on HMTLCollection
            for (var i=0; i<navbarLinks.length; i++){
                // console.log(navbarLinks[i]);
                navbarLinks[i].style.paddingTop = "0.5em"; navbarLinks[i].style.paddingBottom = "0.5em"; navbarLinks[i].style.margin = "0.25em";
            }
    }

    // Return back margins (almost the same, better length measures)
    function restoreMarginNavbar(){
        let navbarLinks = document.getElementsByClassName("navbarLink");
        for (var i=0; i<navbarLinks.length; i++){
            navbarLinks[i].style.paddingTop = "1em"; navbarLinks[i].style.paddingBottom = "1em"; navbarLinks[i].style.margin = "6px";
        }
    }

    // Call functions when the page is loaded for initial placing the elements - repeating the logic from above
    if(visualViewport.width > 1550){
        moveNamePolEl(); makePyramid(shift_y=0, x_reduce_distance = 0);
    } else if((visualViewport.width < 1550) && (visualViewport.width > 1000)){
        makePyramid(shift_y = 25, x_reduce_distance = 0);
    } else if((visualViewport.width < 1000) && (visualViewport.width > 700)){
        makePyramid(shift_y = 25, x_reduce_distance = 60);
    } else if((visualViewport.width < 700) && (visualViewport.width > 500)){
        makePyramid(shift_y = 25, x_reduce_distance = 76); document.querySelector(".flexboxContainer").style.flexDirection = "column";
        makeLessMarginNavbar();
    } else if(visualViewport.width <= 500){
        document.querySelector("#RepresentingZernikeStr").textContent = "Representing a few Zernike profiles in 2 columns";
        document.querySelector(".flexboxContainer").style.flexDirection = "column";
        putImagesInCols(initial_set=true); makeLessMarginNavbar();
    }
});
