(function () {
  var blockquotes = document.querySelectorAll("blockquote.tiktok-embed");
  Array.prototype.forEach.call(blockquotes, function (blockquote) {
    if (blockquote.querySelector("iframe")) return;
    var iframe = document.createElement("iframe");
    iframe.src =
      "https://www.tiktok.com/embed/v2/?id=" +
      encodeURIComponent(blockquote.getAttribute("data-video-id") || "");
    iframe.title = "TikTok video";
    iframe.setAttribute("allowfullscreen", "");
    iframe.style.cssText =
      "width:100%;height:739px;display:block;border:0;max-height:739px";
    blockquote.appendChild(iframe);
  });
})();
