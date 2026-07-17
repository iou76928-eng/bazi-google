const fs = require('fs');
const path = require('path');
const { getHistoricalRates } = require('dukascopy-node');

const outDir = path.resolve('data_bidask');
fs.mkdirSync(outDir, { recursive: true });

async function downloadSide(priceType, outputPath) {
  console.log(`Downloading ${priceType} XAUUSD M1...`);
  const csv = await getHistoricalRates({
    instrument: 'xauusd',
    dates: {
      from: new Date('2023-11-01T00:00:00.000Z'),
      to: new Date('2026-07-17T00:00:00.000Z')
    },
    timeframe: 'm1',
    priceType,
    format: 'csv',
    volumes: true,
    ignoreFlats: true,
    batchSize: 10,
    pauseBetweenBatchesMs: 400,
    retryCount: 5,
    retryOnEmpty: false,
    failAfterRetryCount: true,
    pauseBetweenRetriesMs: 1500,
    useCache: true,
    cacheFolderPath: path.resolve('.dukascopy-cache')
  });
  if (!csv || csv.length < 1000) {
    throw new Error(`${priceType} download returned insufficient data`);
  }
  fs.writeFileSync(outputPath, csv, 'utf8');
  console.log(`${priceType} saved: ${outputPath}, bytes=${Buffer.byteLength(csv)}`);
}

(async () => {
  try {
    await downloadSide('bid', path.join(outDir, 'xauusd_bid_m1.csv'));
    await downloadSide('ask', path.join(outDir, 'xauusd_ask_m1.csv'));
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
})();
