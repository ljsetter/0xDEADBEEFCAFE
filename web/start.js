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
    var script = spawn("python", ["../python/testScript.py", "(XRAY IMAGE"])

    process.stdout.on('data', function(data) {
        res.json(data.toString());
    })
});

app.listen(SERVICE_PORT);