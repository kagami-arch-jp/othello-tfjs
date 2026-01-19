function isBrowser() {
  return typeof document!=='undefined'
}
function isServer() {
  return typeof __SERVER__!=='undefined' && __SERVER__===true
}
function isGitPage() {
  return isBrowser() && location.href.indexOf('github.io/')>-1
}
function defer() {
  let resolve, reject;
  const promise = new Promise((_resolve, _reject) => {
    resolve = _resolve;
    reject = _reject;
  });
  return { promise, resolve, reject };
}
function query(k) {
  return isBrowser() && location.href.match(new RegExp(`\\b${k}=(.+?)\\b|$`))[1] || ''
}
function sleep(t) {
  return new Promise(r=>setTimeout(r, t))
}
function copy(x) {
  if(Array.isArray(x)) {
    return x.map(copy)
  }
  if(typeof x==='object' && x!==null) {
    const o={}
    for(const k in x) {
      o[k]=copy(x[k])
    }
    return o
  }
  return x
}
function assert(x) {
  if(!x) throw new Error(`Uncaught AssertionError [ERR_ASSERTION]: ${x} == true`)
}

const gameName='othello'

function fetchSaveRecord(steps) {
  const sav=JSON.stringify(steps)
  const xhr=new XMLHttpRequest
  xhr.open('POST', `/do?a=save`, true)
  xhr.setRequestHeader('content-type', 'application/x-www-form-urlencoded')
  xhr.send(sav)
}
function $(x) {
  return document.querySelector(x)
}

const [tf, fs]=(_=>{
  if(isBrowser()) {
    return [window.tf, null]
  }
  if(isServer()) {
    return [null, require('fs')]
  }
  return [require('@tensorflow/tfjs-node-gpu'), require('fs')]
})()

const mod_cache={}
async function loadModel(createModel, noCache) {
  if(noCache) {
    for(const c in mod_cache) delete mod_cache[c]
  }
  const [savePath, loadPath]=isBrowser()?
    [null, './mod/model.json?t='+Math.floor(Date.now()/86400e3)]:
    ['file://'+__dirname+'/'+gameName+'/mod', 'file://'+__dirname+'/'+gameName+'/mod/model.json']
  const saveFn=async model=>model.save(savePath)
  try{
    if(!mod_cache[loadPath]) {
      mod_cache[loadPath]=await tf.loadLayersModel(loadPath)
    }
    const model=mod_cache[loadPath]
    return [model, _=>saveFn(model)]
  }catch(e) {
    console.log(e.message)
  }
  const model=createModel()
  await saveFn(model)
  return [model, _=>saveFn(model)]
}


function rotate90(map, r) {
  const map2=[]
  for(let i=0; i<r; i++) {
    for(let j=0; j<r; j++) {
      map2[j]=map2[j] || []
      map2[j][r-1-i]=map[i][j]
    }
  }
  return map2
}
function flip180(map, r, horizon) {
  const map2=[]
  for(let i=0; i<r; i++) {
    for(let j=0; j<r; j++) {
      if(horizon) {
        map2[i]=map2[i] || []
        map2[i][r-j-1]=map[i][j]
      }else{
        map2[r-i-1]=map2[r-i-1] || []
        map2[r-i-1][j]=map[i][j]
      }
    }
  }
  return map2
}
function rotate90_ij(ij, R) {
  const [i, j]=ij
  // 0, 2 => 1, 0  => 3, 1 => 2, 3 => 0, 2
  return [j, R-1-i]
}
function flip180_ij(ij, R, horizon) {
  const [i, j]=ij
  if(horizon) return [i, R-j-1]
  return [R-i-1, j]
}
function rotate90_d(d) {
  return (d+1)%4
}
function flip180_d(d, horizon) {
  if(horizon) {
    if(d===1 || d===3) return 4-d
  }else{
    if(d===0 || d===2) return 2-d
  }
  return d
}
function argu_xy(map, ...ij) {
  assert(map.length===map[0].length)
  map=copy(map)
  const MAP_R=map.length
  function apply_ij(arr_func, num_func, ...arg) {
    for(let i=0; i<ij.length; i++) {
      const a=ij[i]
      ij[i]=(Array.isArray(a)? arr_func: num_func)(a, ...arg)
    }
    return ij
  }
  const ret=[[map, ...ij]]
  for(let n=1; n<4; n++) {
    ret.push([
      map=rotate90(map, MAP_R),
      ...apply_ij(rotate90_ij, rotate90_d, MAP_R),
    ])
  }
  ret.push([
    map=flip180(map, MAP_R, true),
    ...apply_ij(flip180_ij, flip180_d, MAP_R, true),
  ])
  for(let n=1; n<4; n++) {
    ret.push([
      map=rotate90(map, MAP_R),
      ...apply_ij(rotate90_ij, rotate90_d, MAP_R),
    ])
  }
  return ret
}

/*
const r=argu_xy([
  [0, 2, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
], [0, 0], [0, 3])
for(const [m, [i, j], [i1, j1]] of r) {
  m[i][j]='x'
  m[i1][j1]='y'
  console.log(m.map(c=>c.join('')).join('\n'), '\n')
}
process.exit()
*/

function getRecordJSONList(sort) {
  try{
    const exp_dir=__dirname+'/'+gameName+'/tracks'
    const arr=fs.readdirSync(exp_dir).filter(x=>x.indexOf('.json')>-1)
    if(sort) {
      arr.sort((a, b)=>{
        const n=x=>{
          const [, n0, n1]=x.match(/(?:test-(?:[^-]+)-(\d+)-){0,1}(\d+)\.json$/)
          return parseInt(n1)+parseInt(n0)/1e8
        }
        return n(b)-n(a)
      })
    }
    return arr.map(fn=>`${exp_dir}/${fn}`)
  }catch(e) {
    return []
  }
}

function removeTestRecords() {
  try{
    getRecordJSONList().forEach(fn=>{
      if(isHumanRecord(fn)) {
        fs.writeFileSync(toNormalRecord(fn), fs.readFileSync(fn))
      }
      fs.unlinkSync(fn)
    })
  }catch(e) {}
}

function removeDuplicateRecord(toKey) {
  console.log('remove duplicate steps..')
  const caches=new Set
  let rm=0

  for(const fn of getRecordJSONList(true)) {
    const steps=JSON.parse(fs.readFileSync(fn, 'utf8'))
    const clean_steps=[]
    for(const step of steps) {
      const [map, ij]=step
      let duplicate=false
      const keys=argu_xy(map, ij).map(([v, pos])=>{
        return toKey(v, pos)
      })
      for(const key of keys) {
        if(caches.has(key)) {
          duplicate=true
          break
        }
      }
      for(const key of keys) {
        caches.add(key)
      }
      if(!duplicate) {
        clean_steps.push(step)
      }else{
        rm++
      }
    }
    if(clean_steps.length!==steps.length) {
      if(clean_steps.length>0) {
        fs.writeFileSync(fn, JSON.stringify(clean_steps))
      }else try{
        fs.unlinkSync(fn)
      }catch(e) {}
    }
  }
  if(rm>0) console.log(`-- ${rm} duplicate steps removed --`)
}
//removeDuplicateRecord()
//process.exit()

let rec_i=0
function nodeSaveRecord(steps, isByHuman) {
  const exp_dir=__dirname+'/'+gameName+'/tracks'
  try{
    fs.mkdirSync(exp_dir)
  }catch(e) {}
  const fn=`${exp_dir}/${isByHuman? 'expert-': ('test-'+(process.env.isStepIndex || 'x')+'-'+(rec_i++)+'-')}${Date.now()}.json`
  fs.writeFileSync(fn, typeof steps==='string'? steps: JSON.stringify(steps))
}

function isHumanRecord(fn) {
  return !!fn.match(/\bexpert-\d/)
}

function toNormalRecord(fn) {
  return fn.replace(/\bexpert-(\d)/, '$1')
}

function rsort(arr) {
  return arr.sort(_=>Math.random()-.5)
}

function generateExpertTracks(toTrainData, maxFilePerIter=100) {
  const fnList=getRecordJSONList()
  const fnCache={}
  function loadFile(fn) {
    fnCache[fn]=fnCache[fn] || fs.readFileSync(fn, 'utf8')
    return JSON.parse(fnCache[fn])
  }
  function loadXY(fn, track_steps) {
    const res=loadFile(fn)
    for(const [map, ij, valid_ij_arr=[]] of res) {
      const caches=new Set

      /*
      const e=argu_xy(map, ij, ...valid_ij_arr)
      for(const [_map, ...c] of e) {
        console.log(_map.join('\n'), c[0], ['up','right','down','left'][c[0]])
      }
        process.exit()*/

      for(const [_map, _ij, ..._valid_ij_arr] of argu_xy(map, ij, ...valid_ij_arr)) {
        const key=_map.map(c=>c.join('')).join('')
        if(caches.has(key)) continue
        caches.add(key)
        track_steps.push(toTrainData(_map, _ij, _valid_ij_arr))
      }
    }
  }
  function *iterator() {
    let track_steps=[], files=0
    for(;;) {
      rsort(fnList)
      for(let fn of fnList) {
        loadXY(fn, track_steps)
        files++
        if(files%maxFilePerIter===0) {
          yield rsort(track_steps)
          track_steps=[]
          files=0
        }
      }
    }
  }
  const steps=fnList.reduce((n, fn)=>{
    return n+loadFile(fn).length*8
  }, 0)
  return [steps, iterator()]
}

function createTrainData(batchSize, toBatchData, ...arg) {
  const [exp_len, exp_iter]=generateExpertTracks(...arg)
  function *iterator() {
    for(let xy=[];;) {
      if(xy.length>batchSize) {
        const xys=xy.splice(0, batchSize)
        yield toBatchData(xys)
      }else{
        xy=xy.concat(exp_iter.next().value)
      }
    }
  }
  return [iterator(), Math.ceil(exp_len/batchSize)]
}

function agentBatchPredict(model) {
  let dt=0
  const inps=[], cbs=[]
  function doPredict(...inp) {
    const def=defer()
    for(let i=0; i<inp.length; i++) {
      inps[i]=inps[i] || []
      inps[i].push(inp[i])
    }
    cbs.push(def.resolve)
    clearImmediate(dt)
    dt=setImmediate(_=>{
      const res=predict(model, ...inps)
      res.map((d, i)=>cbs[i](d))
      inps.map(x=>x.splice(0))
      cbs.splice(0)
    })
    return def.promise
  }
  return doPredict
}

async function callGenerateTracks(isStep, index, argv) {
  const {spawn}=require('child_process')
  const prog=spawn('npm', ['run', 'generate:'+gameName], {
    stdio: isStep? 'ignore': ['ignore', process.stdout, process.stdout],
    env: {
      ...process.env,
      isStepIndex: index,
      isStep: isStep? 1: 0,
      isStepArgv: argv? JSON.stringify(argv): undefined,
    },
  })
  await new Promise(r=>{
    prog.on('close', r)
  })
}

async function main({
  browserFunc,
  testFunc,
  generateConfig,
  map2key,
  createModelFunc,
  createTrainDataFunc,
  testAfterTrainDone,
  lr=2e-4,
  epochsPerDataset=5,
  lossFunc,

}) {
  const [model, save]=await loadModel(createModelFunc)

  const task=!isBrowser() && (_=>{
    // if('@TRAIN'==='true') return 'train'
    if('@TEST'==='true') return 'test'
    if('@GENERATE'==='true') return 'generate'
    return 'train'
  })()

  if(isBrowser()) {
    browserFunc(model)
  }else if(task==='test') {
    testFunc(model)
  }else if(task==='generate') {
    const {isStep, isStepArgv}=process.env
    let {min, fn, getArgv}=generateConfig
    if(isStep=='1') {
      await fn(model, isStepArgv? JSON.parse(isStepArgv): undefined)
    }else{
      const max_subprocess=require('os').cpus().length || 1
      const _argv=getArgv? await getArgv(model): undefined
      const e={tasks: 0, end: false, i: 0, arr: []}
      let count=0, start_t=Date.now()
      let logs=0
      const init_n=getRecordJSONList().length || 0
      const log=_=>{
        const _count=getRecordJSONList().length
        if(_count<=count) return false
        const per_second=((_count-init_n+1e-8)/((Date.now()-start_t)/1e3+1e-8)).toFixed(3)
        const str=`+ ${_count}/${Math.max(min, _count)} records, ${per_second} per second    `
        process.stdout.write('\b'.repeat(logs)+str)
        logs=str.length
        count=_count
        return count>=min
      }
      if(log()) {
        console.log()
        return
      }
      for(; !e.end;) {
        if(e.tasks>=max_subprocess) {
          await sleep(500)
          continue
        }
        const n=getRecordJSONList().length
        e.tasks++
        e.i++
        const task=callGenerateTracks(true, e.i, _argv)
        e.arr.push(task)
        task.then(_=>{
          e.tasks--
          if(log()) e.end=true
        })
      }
      await Promise.all(e.arr)
      log()
      console.log()
    }
  }else{
    model.summary()
    //process.exit()
    for(;;) {
      const opt=tf.train.adam(lr)
      model.compile({
        optimizer: opt,
        loss: (yTrue, yPred)=>{
          const loss=lossFunc(yTrue, yPred)
          if(loss.isNaN().dataSync().indexOf(1)>-1) {
            console.log("-- loss is NaN, early stopped --")
            process.exit()
          }
          return loss
        },
        metrics: ['acc'],
      })

      await callGenerateTracks()
      removeDuplicateRecord(map2key)

      const dataset=createTrainDataFunc()
      const [ds, steps]=dataset

      await model.fitDataset({iterator: _=>ds}, {
        epochs: epochsPerDataset,
        batchesPerEpoch: steps,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            save()
            console.log('-- model updated --')
          }
        },
      })

      removeTestRecords()

    }
  }
}
