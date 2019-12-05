function postResults(data, clone){
    var content = document.getElementById("results-row").content;
    var clone = document.importNode(content, true);
    file = document.getElementById("photo").files[0].name
    clone.querySelector("h3").textContent += file
    data.json().then(body => {
        attributes = body.split(' ')
        path = attributes[0].replace('public', '');
        clone.querySelector("img").src = path
        clone.querySelector("h4").textContent += attributes[1];
        document.getElementById("result-block").prepend(clone);

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