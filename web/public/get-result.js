function postResults(data){
    var content = document.getElementById("results-row").content;
    var clone = document.importNode(content, true);
    data.json().then(body => {
        clone.querySelector("h4").textContent += body;
        document.getElementById("result-block").appendChild(clone);

    })
}

function onUpload(){
    fetch("/api/process-image", {
        body: new FormData(document.getElementById("upload-form")),
        method: "post"
    }).then(postResults);
}

document.getElementById("upload-form").onsubmit = function () {
    onUpload();
    return false;
}