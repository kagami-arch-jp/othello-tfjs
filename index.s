<?js

define('__SERVER__', true)

const fs=require('fs')
const path=require('path')
const qs=require('querystring')
const url=require('url')

const lib=include_js('./lib.js')
lib.setGameName('othello')

const {query}=url.parse($_RAW_REQUEST.url)
const q=qs.parse(query)

function fsResponse(fn) {
	const {ext}=path.parse(fn)
	setStatus(200)
  setResponseHeaders({
		'content-type': ({
			'.js': 'text/javascript',
			'.css': 'text/css',
			'.png': 'image/png',
			'.jpg': 'image/jpeg',
			'.html': 'text/html',
		})[ext] || 'application/octet-stream',
	})
	return new Promise(r=>{
		fs.createReadStream(fn).
		  on('close', r).
		  on('error', e=>{
  			setStatus(500)
	  		setResponseHeaders({
		  		'content-type': 'text/html'
			  })
  			echo(`<h1>${e.message}</h1>`)
	  	}).
	  	on('ready', _=>{
      	flushHeader()
	  	}).
	  	pipe($_RAW_RESPONSE)
	})
}

function data() {
  return new Promise(resolve=>{
    const buf=[]
    $_RAW_REQUEST.on('data', b=>{
	    buf.push(b)
    })
    $_RAW_REQUEST.on('end', _=>{
      resolve(Buffer.concat(buf).toString())
    })
  })
}

const file=path.normalize($_RAW_REQUEST.url).replace(/\?.*/, '')
const gameName=JSON.parse(process.env.npm_config_argv).original[1].match(/:(.+)/)[1]

if(file==='/') {
	setStatus(302)
	setResponseHeaders({
		location: `/${gameName}/index.html`,
	})
}else if(file==='/do') {
	if(q.a==='save') {
  	Sync.Push(data().then(steps=>{
			lib.nodeSaveRecord(steps, true)
		}))
  }
}else{
  const fn=path.resolve(__dirname+'/'+file)
  Sync.Push(fsResponse(fn))
}
