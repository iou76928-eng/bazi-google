const fs = require('fs');
const path = require('path');
const { getHistoricalRates } = require('dukascopy-node');

const outDir = path.resolve('data_xau_h4');
fs.mkdirSync(outDir, { recursive: true });

async function download(priceType) {
  const target = path.join(outDir, `xauusd_${priceType}_h4.csv`);
  const csv = await getHistoricalRates({
    instrument: 'xauusd',
    dates: {
      from: new Date('2010-01-01T00:00:00.000Z'),
      to: new Date('2026-07-17T00:00:00.000Z')
    },
    timeframe: 'h4',
    priceType,
    format: 'csv',
    volumes: true,
    ignoreFlats: true,
    batchSize: 30,
    pauseBetweenBatchesMs: 200,
    retryCount: 5,
    retryOnEmpty: false,
    failAfterRetryCount: true,
    pauseBetweenRetriesMs: 1200,
    useCache: true,
    cacheFolderPath: path.resolve('.dukascopy-cache')
  });
  if (!csv || csv.length < 1000) throw new Error(`${priceType} data insufficient`);
  fs.writeFileSync(target, csv, 'utf8');
  console.log(`${priceType}: ${Buffer.byteLength(csv)} bytes`);
}

(async () => {
  try {
    await download('bid');
    await download('ask');
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
})();
