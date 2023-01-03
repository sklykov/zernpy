// the way of adding the Event listener - to guarantee that the page and all elements are available
document.addEventListener("DOMContentLoaded", function () {
    console.log("Page loaded");
    let selector = document.getElementsByName("Index type")[0];

    // register event for tracing the new selected value
    selector.addEventListener("change", () => {
        console.log(selector.value);
    });

});