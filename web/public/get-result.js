function postResults(data){
    console.log(data);
    var content = document.getElementById("results-row").content;
    var clone = document.importNode(content, true);
    data.json().then(body => {
        attributes = body.split(' ')
        path = attributes[0].replace('public', '');
        clone.querySelector("img").src = path
        clone.querySelector("h4").textContent += attributes[1];
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