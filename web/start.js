const express = require('express');
const http = require('http');
const spawn = require('child_process').spawn;
const async = require('express-async-await');
const fetch = require('node-fetch');

const SERVICE_PORT = 4701;

app = express();
server = http.createServer(app);

app.use(express.static(__dirname + '/public'));

app.get('/', function(req, res){
    res.sendFile(directory + '/public/index.html');
});

app.post('/api/process-image', async function(req, res){
    /* Image should be processed here */
    res.json(":)");
});

app.get('/api/process-image', async function(req, res){
    res.send(":(");
})

app.listen(SERVICE_PORT);