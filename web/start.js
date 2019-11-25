const express = require('express');
const http = require('http');
const spawn = require('child_process').spawn;
const async = require('express-async-await');
const fetch = require('node-fetch');
const multer = require('multer');
const bodyParser = require('body-parser');
const upload = multer({ storage: multer.memoryStorage() })

const SERVICE_PORT = 4701;

app = express();
server = http.createServer(app);

app.use(express.static(__dirname + '/public'));

app.get('/', function(req, res){
    res.sendFile(directory + '/public/index.html');
});

app.post('/api/process-image', bodyParser.urlencoded({ extended: true }), upload.single('photo'), async function(req, res){
    console.log(req.file);
    

    var script = spawn("python", ["./predict.py", "person1671_virus_2887.jpeg"])

    script.stdout.on('data', function(data) {
        res.json(data.toString());
    })

    script.stderr.on('data', function(data){
        console.log(data.toString());
    })
});

app.listen(SERVICE_PORT);