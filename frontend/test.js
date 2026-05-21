import http from 'http';

http.get('http://localhost:5173', (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => console.log("FETCHED SERVER HTML OK"));
}).on('error', (e) => console.error(e));
