$(document).ready(function(){
  // Make headers real interactive elements: focusable, announced as buttons,
  // and visible to link-hint tools (Vimium etc.) that scan for role/tabindex.
  $(".collapsibleheader")
    .attr({ role: "button", tabindex: 0, "aria-expanded": "false" })
    .on("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        $(this).trigger("click");
      }
    });

  $(".collapsibleheader").click(function () {
      $collapsibleheader = $(this);
      //getting the next element
      $collapsiblecontent = $collapsibleheader.next();
      //open up the collapsiblecontent needed - toggle the slide- if visible, slide up, if not slidedown.
      $collapsiblecontent.slideToggle(500, function () {
          //execute this after slideToggle is done
          var visible = $collapsiblecontent.is(":visible");
          $collapsibleheader.attr("aria-expanded", visible ? "true" : "false");
          //change text of collapsibleheader based on visibility of collapsiblecontent div
          $collapsibleheader.find("span:first-child").text(function () {
              //change text based on condition
              return visible ? "- " : "+ ";
          });
      });

  });
});
