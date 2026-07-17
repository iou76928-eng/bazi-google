const fs = require('fs');
const path = require('path');
const { getHistoricalRates } = require('dukascopy-node');

const instruments = ['xauusd', 'eurusd', 'btcusd'];
const outDir = path.resolve('data_universal');
fs.mkdirSync(outDir, { recursive: true });

async function download(instrument, priceType) {
  const target = path.join(outDir, `${instrument}_${priceType}_m5.csv`);
  console.log(`Downloading ${instrument} ${priceType} M5...`);
  const csv = await getHistoricalRates({
    instrument,
    dates: {
      from: new Date('2023-11-01T00:00:00.000Z'),
      to: new Date('2026-07-17T00:00:00.000Z')
    },
    timeframe: 'm5',
    priceType,
    format: 'csv',
    volumes: true,
    ignoreFlats: true,
    batchSize: 20,
    pauseBetweenBatchesMs: 250,
    retryCount: 5,
    retryOnEmpty: false,
    failAfterRetryCount: true,
    pauseBetweenRetriesMs: 1200,
    useCache: true,
    cacheFolderPath: path.resolve('.dukascopy-cache')
  });
  if (!csv || csv.length < 1000) throw new Error(`${instrument} ${priceType} insufficient data`);
  fs.writeFileSync(target, csv, 'utf8');
  console.log(`Saved ${target}: ${Buffer.byteLength(csv)} bytes`);
}

(async () => {
  try {
    for (const instrument of instruments) {
      await download(instrument, 'bid');
      await download(instrument, 'ask');
    }
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
})();
