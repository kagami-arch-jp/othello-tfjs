const MAP_R=8
const POS_BLACK=1 // black move or next move
const POS_WHITE=2 // white move

function findDirections(map, i, j, R, u, firstOnly) {
  const dirs=[]
  for(let x=0, q=0; x<3; x++) {
    for(let y=0; y<3; y++) {
      if(x===1 && y===1) continue
      const _x=x-1, _y=y-1
      const _u=3-u
      for(let _i=i, _j=j, n=0, t=0; ;n++) {
        _i+=_x
        _j+=_y
        if(_i<0 || _j<0 || _i>=R || _j>=R) break
        const _c=map[_i][_j]
        if(_c===_u) {
          t++
        }else if(_c===u) {
          if(t>0) {
            dirs.push([_x, _y, t])
            if(firstOnly) q=1
            break
          }else break
        }else if(_c===0) break
      }
      if(q) break
    }
    if(q) break
  }
  return dirs
}
function getValidPositionArr(map, R, u) {
  const posArr=[]
  for(let i=0; i<R; i++) {
    for(let j=0; j<R; j++) {
      if(map[i][j]!==0) continue
      const dirs=findDirections(map, i, j, R, u, 1)
      if(dirs.length>0) posArr.push([i, j])
    }
  }
  return posArr
}
function doNextStep(map, R, u, i, j) {
  map[i][j]=u
  const dirs=findDirections(map, i, j, R, u, 0)
  for(let [di, dj, t] of dirs) {
    for(let _t=1; _t<=t; _t++) {
      map[i+di*_t][j+dj*_t]=u
    }
  }
}
function log(map, validPosArr=[]) {
  for(let [i, j] of validPosArr) map[i][j]='x'
  console.log(map.map(x=>x.join(' ')).join('\n'), '\n')
}
function newMap(emptyMap) {
  const ret=[]
  for(let i=0; i<MAP_R; i++) {
    ret[i]=[]
    for(let j=0; j<MAP_R; j++) {
      ret[i][j]=0
    }
  }
  if(!emptyMap) {
    ret[MAP_R/2-1][MAP_R/2-1]=POS_BLACK
    ret[MAP_R/2-1][MAP_R/2]=POS_WHITE
    ret[MAP_R/2][MAP_R/2-1]=POS_WHITE
    ret[MAP_R/2][MAP_R/2]=POS_BLACK
  }
  return ret
}
function newRoundModel() {
  const map=newMap()
  let nextPlayer=POS_BLACK
  let gameover=0

  const customSteps=[]

  function isGameover() {
    return gameover
  }

  function changePlayer() {
    nextPlayer=POS_BLACK+POS_WHITE-nextPlayer
  }

  function getPlayer(returnId) {
    if(returnId) return nextPlayer
    return nextPlayer===POS_BLACK? 'black': 'white'
  }

  function doStep(fn, record) {
    let validPosArr=getValidArr()
    if(!validPosArr.length) {
      changePlayer()
      validPosArr=getValidArr()
    }
    if(!validPosArr.length) {
      gameover=1
      return
    }
    const _i=validPosArr.length===1? 0: fn(map, validPosArr)
    if(_i<0) return;
    const [i, j]=validPosArr[_i]
    if(record && validPosArr.length>1) {
      customSteps.push([mask_map(map, nextPlayer), [i, j], validPosArr])
    }
    doNextStep(map, MAP_R, nextPlayer, i, j)
    const canCurrMove=getValidArr().length>0
    changePlayer()
    const canNextMove=getValidArr().length>0
    if(canNextMove) return;
    if(canCurrMove) {
      changePlayer()
      return
    }
    gameover=1
  }

  function posArr2Mask(validPosArr) {
    const mask=newMap(true)
    for(const [i, j] of validPosArr) {
      mask[i][j]=1
    }
    return mask
  }

  function getState() {
    let bn=0, wn=0
    for(let i=0; i<MAP_R; i++) {
      for(let j=0; j<MAP_R; j++) {
        const c=map[i][j]
        if(c===POS_BLACK) bn++
        else if(c===POS_WHITE) wn++
      }
    }
    return [bn, wn]
  }

  function getValidArr(toMask) {
    const t=getValidPositionArr(map, MAP_R, nextPlayer)
    return toMask? [t, posArr2Mask(t)]: t
  }

  return {
    isGameover,
    doStep,
    getPlayer,
    getState,
    getValidArr,
    posArr2Mask,
    map,
    customSteps,
  }
}

function initPage(map, agentPlayer) {
  let str=`<div class="map">`+
  map.map((li, i)=>`<div class="line">`+
    li.map((b, j)=>`<div data-pos='i${i}j${j}'></div>`).join('')+
  '</div>').join('')+
  `</div>`
  str+=`<div class="state">
    <div class="li">
      <div class="icon block black"></div>
      <div class="num ${agentPlayer===POS_BLACK? 'agent': ''}" id="bn">2</div>
    </div>
    <div class="li">
      <div class="icon block white"></div>
      <div class="num ${agentPlayer===POS_WHITE? 'agent': ''}" id="wn">2</div>
    </div>
  </div>`
  const n=agentPlayer===POS_BLACK? 1: 2
  const txt=n===1? 'Player First': 'Agent First'
  str+=`<div class="btns">
    <a href="?f=${n}">${txt}</a>
    <a href="?f=${3-n}&r=${Date.now()}">New Game</a>
  </div>`
  app.innerHTML=str
}
function renderPage(map, validPosArr=[], nextPlayer=0) {
  let bn=0, wn=0
  map.map((li, i)=>{
    li.map((b, j)=>{
      if(b===POS_BLACK) bn++
      else if(b===POS_WHITE) wn++
      $(`[data-pos="i${i}j${j}"]`).className=`block ${
        ['empty', 'black', 'white'][b]
      } ${
        validPosArr.find(([_i, _j])=>_i===i && _j===j)? 'next next-'+['', 'black', 'white'][nextPlayer]: ''
      }`
    })
  })
  $('.num#bn').innerHTML=bn
  $('.num#wn').innerHTML=wn
}

function predict(model, maps, ns, validLs, outputMasks) {
  const x=tf.tidy(_=>tf.tensor3d(maps.map((map, k)=>{
    return mask_map(map, ns[k])
  }), [maps.length, MAP_R, MAP_R]))
  const mask=tf.tidy(_=>tf.tensor3d(outputMasks, [outputMasks.length, MAP_R, MAP_R]))

  const p1=model.predictOnBatch([x, mask])
  const {indices}=tf.topk(p1.reshape([maps.length, MAP_R * MAP_R]))
  const idx=indices.dataSync()

  return [...idx].map((v, n)=>{
    const j=v%MAP_R
    const i=(v-j)/MAP_R
    return validLs[n].findIndex(a=>a[0]===i && a[1]===j)
  })
}

function mask_map(map, next) {
  const m=copy(map)
  for(let li of m) {
    for(let i=0; i<li.length; i++) {
      if(li[i]===0) continue
      li[i]=li[i]===next? POS_BLACK: POS_WHITE
    }
  }
  return m
}

if(isBrowser()) main({
  browserFunc: async model=>{
    const agentFirst=query('f')=='2'
    const customPlayer=agentFirst? 'white': 'black'
    const g=newRoundModel()
    const render=_=>{
      renderPage(g.map, g.getValidArr(), g.getPlayer(true))
    }

    initPage(g.map, agentFirst? POS_BLACK: POS_WHITE)
    document.addEventListener('click', e=>{
      if(e.target.className.indexOf('next')<0) return;
      const [i, j]=e.target.getAttribute('data-pos').match(/\d/g)
      if(g.getPlayer()!==customPlayer) return;
      g.doStep((map, validArr)=>validArr.findIndex(x=>x[0]===+i && x[1]===+j), true)
      render()
    })
    render()

    for(let randomStep=1; !g.isGameover(); ) {
      await sleep(1e2)
      if(g.getPlayer()===customPlayer) continue
      await sleep(2e2)
      g.doStep((map, validArr)=>{
        const m=g.posArr2Mask(validArr)
        if(randomStep>0) {
          randomStep--
          return Math.floor(Math.random()*validArr.length)
        }
        return predict(model, [map], [agentFirst? POS_BLACK: POS_WHITE], [validArr], [m])[0]
      })
      render()
    }

    const [bn, wn]=g.getState()
    let alert_msg='draw'
    if(bn===wn) {
      alert_msg='draw'
    }else if((agentFirst&&bn>wn) || (!agentFirst&&bn<wn)) {
      alert_msg='agent won!'
    }else{
      alert_msg=isGitPage()? 'you won!': 'record saved'
      fetchSaveRecord(g.customSteps)
    }

    await sleep(5e2)
    alert(alert_msg)

  }
})
