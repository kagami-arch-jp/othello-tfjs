#include "../lib.js"
#include "./game.js"

function createModel() {
  const input=tf.input({shape: [MAP_R, MAP_R]})
  const outputMask=tf.input({shape: [MAP_R, MAP_R]})
  let x=input

  x=tf.layers.reshape({targetShape: [MAP_R, MAP_R, 1]}).apply(x)
  x=tf.layers.conv2d({kernelSize: 3, filters: 32, padding: 'same', activation: 'relu'}).apply(x)
  x=tf.layers.maxPooling2d({poolSize: 2, strides: 2}).apply(x)
  x=tf.layers.conv2d({kernelSize: 3, filters: 64, padding: 'same', activation: 'relu'}).apply(x)
  x=tf.layers.maxPooling2d({poolSize: 2, strides: 2}).apply(x)
  x=tf.layers.conv2d({kernelSize: 3, filters: 128, padding: 'same', activation: 'relu'}).apply(x)
  x=tf.layers.maxPooling2d({poolSize: 2, strides: 2}).apply(x)
  x=tf.layers.flatten({}).apply(x)
  x=tf.layers.dense({units: 256, activation: 'relu'}).apply(x)
  x=tf.layers.dense({units: MAP_R*MAP_R, activation: 'softmax'}).apply(x)
  x=tf.layers.reshape({targetShape: [MAP_R, MAP_R]}).apply(x)
  x=tf.layers.multiply().apply([x, outputMask])
  const model=tf.model({inputs: [input, outputMask], outputs: x})

  return model
}

function getTrainData(batchSize) {
  const g=newRoundModel()
  return createTrainData(batchSize, xys=>{
    const xs=[], masks=[], ys=[], indices=[]
    for(const {x, mask, validIndices, y} of xys) {
      xs.push(x)
      masks.push(mask)
      ys.push(y)
    }

    return {
      xs: [
        tf.tidy(_=>tf.tensor3d(xs, [batchSize, MAP_R, MAP_R], 'int32')),
        tf.tidy(_=>tf.tensor3d(masks, [batchSize, MAP_R, MAP_R], 'int32')),
      ],
      ys: tf.tidy(_=>tf.tensor3d(ys, [batchSize, MAP_R, MAP_R], 'int32')),
    }

  }, (_map, _ij, _valid_ij_arr)=>{
    return {
      x: _map,
      mask: g.posArr2Mask(_valid_ij_arr),
      y: g.posArr2Mask([_ij]),
    }
  })
}

async function generateTracks(model, {
  n=100,
  isAgentFirst=false,
  random=false,
  isFromTest=false,
}, isTraining) {

  let agentWins=0, randomWins=0, drawRound=0

  const doPredict=agentBatchPredict(model)

  async function newRound() {
    const g=newRoundModel()
    const LEARN_FROM_AGENT_RATIO=Math.random()<.9? 9/32: 0
    for(let randomStep=1; !g.isGameover();) {
      const isAgentStep=g.getPlayer()===(isAgentFirst? 'black': 'white')
      function random_step(record=false) {
        g.doStep((_, validArr)=>{
          return Math.floor(Math.random()*validArr.length)
        }, record)
      }
      async function agent_step(POS, record=false) {
        const [t, m]=g.getValidArr(true)
        const p=await doPredict(g.map, POS, t, m)
        g.doStep(_=>p, record)
      }
      if(random) {
        random_step()
      }else{
        if(isAgentStep) {
          if(randomStep>0) {
            randomStep--
            random_step()
          }else{
            await agent_step(isAgentFirst? POS_BLACK: POS_WHITE, false)
          }
        }else{
          if(isTraining && Math.random()<LEARN_FROM_AGENT_RATIO) {
            await agent_step(isAgentFirst? POS_WHITE: POS_BLACK, true)
          }else{
            random_step(true)
          }
        }
      }
    }

    const [bn, wn]=g.getState()
    const isAgentWin=(isAgentFirst && bn>wn) || (!isAgentFirst && bn<wn)
    const isDraw=bn===wn

    if(isDraw) {
      drawRound++
    }else if(isAgentWin) {
      agentWins++
    }else{
      randomWins++
    }

    if((isTraining || isFromTest) && g.customSteps && !isAgentWin && !isDraw) {
      nodeSaveRecord(g.customSteps, isFromTest)
    }
  }

  const rounds=[]
  for(let i=0; i<n; i++) {
    rounds.push(newRound())
  }
  await Promise.all(rounds)

  return {
    agentWins,
    randomWins,
    drawRound,
  }

}

if(!isBrowser()) main({
  epochsPerDataset: 5,

  testFunc: async model=>{
    const n=5000
    console.log('test '+n*2+' rounds..')
    const agentFirst=await generateTracks(model, {n, isAgentFirst: true, isFromTest: true})
    const agentSecond=await generateTracks(model, {n, isAgentFirst: false, isFromTest: true})
    const randomFirst=await generateTracks(null, {n, isAgentFirst: true, random: true})
    const randomSecond=await generateTracks(null, {n, isAgentFirst: false, random: true})

    const v=x=>(x/n*100).toFixed(1)+'%'
    const toRet=x=>({
      win: v(x.agentWins),
      draw: v(x.drawRound),
    })

    console.table({
      'first move': {
        agent: toRet(agentFirst),
        random: toRet(randomFirst),
      },
      'second move': {
        agent: toRet(agentSecond),
        random: toRet(randomSecond),
      },
    })
  },

  generateConfig: {
    min: 5000,
    fn: async model=>{
      const n=5000
      const [agentFirst, agentSecond]=await Promise.all([
        generateTracks(model, {n, isAgentFirst: true}, true),
        generateTracks(model, {n, isAgentFirst: false}, true),
      ])
      return agentFirst.randomWins+agentSecond.randomWins
    },
  },

  map2key: (map, [i, j])=>{
    const v=map.map(x=>x.join('')).join('')
    const key1=v.replace(/0+/g, _=>`a${_.length}b`)
    const key2=[v.substr(0, 24), v.substr(24, 48), v.substr(48, 64)].map(x=>parseInt(x, 3).toString(36)).join('_')
    return i+''+j+'_'+(key1.length>key2.length? key2: key1)
  },

  createModelFunc: createModel,
  createTrainDataFunc: _=>getTrainData(2048),

  lossFunc: (yTrue, yPred)=>{
    const loss=tf.metrics.categoricalCrossentropy(
      yTrue.reshape([yTrue.shape[0], MAP_R*MAP_R]),
      yPred.reshape([yTrue.shape[0], MAP_R*MAP_R]),
    )
    return loss
  },
})
