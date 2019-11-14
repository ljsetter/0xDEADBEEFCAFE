const express = require('express');
const http = require('http');
const spawn = require('child_process').spawn;

const SERVICE_PORT = 4701;

app = express();
server = http.createServer(app);

app.use(express.static(__dirname + '/public'));

app.get('/', function(req, res){
    res.sendFile(directory + '/public/index.html')
});

app.get('/process-image', function(req, res){
    /* Image should be processed here */
    res.send(":)");
});

app.listen(SERVICE_PORT);