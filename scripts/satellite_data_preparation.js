/**
 * Google Earth Engine script for satellite data preparation for biomass prediction.
 * 
 * This script creates a multi-sensor stack combining:
 * - Sentinel-2 optical data (seasonal composites)
 * - Landsat 8 optical data (seasonal composites)
 * - Sentinel-1 SAR data
 * - ALOS PALSAR SAR data
 * - Canopy Height
 * - Digital Elevation Model and derived slope
 * 
 * Author: najahpokkiri
 * Date: 2025-05-30
 */

// Load biomass reference data
var image = ee.Image('projects/ee-najah/assets/01_Betul_AGB40');
Map.addLayer(image, {}, 'AGB Reference');

// Define area of interest
var geometry2 = image.geometry(); // Using the biomass image's geometry as our study area

// Define time steps for seasonal composites
var timeSteps = [
  {'start': '2020-01-01', 'end': '2020-05-01'}, // Winter/Spring
  {'start': '2020-05-02', 'end': '2020-09-01'}, // Summer
  {'start': '2020-09-02', 'end': '2021-01-01'}  // Fall/Winter
];

// Center map on our study area
Map.centerObject(geometry2);

//=============================================================================
// 1. SENTINEL-2 OPTICAL DATA
//=============================================================================

// Filter Sentinel-2 collection
var s2Collection = ee.ImageCollection("COPERNICUS/S2_SR")
  .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .filterBounds(geometry2);

// Function to get median image for a given time step
var getMedianImage = function(timeStep) {
  return s2Collection.filter(ee.Filter.date(timeStep.start, timeStep.end)).median();
};

// Get median images for each time step
var s2ImageList = timeSteps.map(getMedianImage);

// Select Sentinel-2 bands ≤ 40 meters resolution
var s2BandsToSelect = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

// Rename the bands for each time step
var renamedS2Images = s2ImageList.map(function(image, index) {
  var suffix = '_T' + (index + 1);
  var newBandNames = s2BandsToSelect.map(function(band) {
    return band + suffix;
  });
  return image.select(s2BandsToSelect).rename(newBandNames);
});

// Merge all seasonal images into one stack
var mergedStackS2 = ee.Image.cat(renamedS2Images);

// Add to map for visualization
var s2VisParams = { bands: ['B4_T3', 'B3_T3', 'B2_T3'], min: 0, max: 3000 };
Map.addLayer(mergedStackS2, s2VisParams, 'Sentinel-2 Seasonal Stack');

//=============================================================================
// 2. LANDSAT 8 OPTICAL DATA
//=============================================================================

// Filter Landsat 8 collection
var l8Collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filter(ee.Filter.date('2019-01-01', '2022-01-01'))
  .filter(ee.Filter.lt('CLOUD_COVER', 30))
  .filterBounds(geometry2);

// Function to get median image for a given time step
var getMedianL8Image = function(timeStep) {
  return l8Collection.filter(ee.Filter.date(timeStep.start, timeStep.end)).median();
};

// Get median images for each time step
var l8ImageList = timeSteps.map(getMedianL8Image);

// Select Landsat 8 surface reflectance bands
var l8BandsToSelect = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];

// Rename the bands for each time step
var renamedL8Images = l8ImageList.map(function(image, index) {
  var suffix = '_T' + (index + 1);
  var newBandNames = l8BandsToSelect.map(function(band) {
    return band + suffix;
  });
  return image.select(l8BandsToSelect).rename(newBandNames);
});

// Merge all seasonal images into one stack
var mergedStackL8 = ee.Image.cat(renamedL8Images);

// Add to map for visualization
var l8VisParams = { bands: ['SR_B4_T2', 'SR_B3_T2', 'SR_B2_T2'], min: 7193, max: 17072 };
Map.addLayer(mergedStackL8, l8VisParams, 'Landsat-8 Seasonal Stack');

//=============================================================================
// 3. SENTINEL-1 SAR DATA
//=============================================================================

// Filter Sentinel-1 collection
var s1Collection = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filterBounds(geometry2);

// Processing function for Sentinel-1
var processS1 = function(image) {
  var vv = image.select('VV');
  var vh = image.select('VH');
  
  return ee.Image.cat([vv, vh])
    .rename(['VV', 'VH'])
    .copyProperties(image, ['system:time_start']);
};

// Function to get median S1 image for a given time step
var getMedianS1Image = function(timeStep) {
  return s1Collection
    .filter(ee.Filter.date(timeStep.start, timeStep.end))
    .map(processS1)
    .median();
};

// Get median S1 images for each time step
var s1ImageList = timeSteps.map(getMedianS1Image);

// Rename S1 bands for each time step
var renamedS1Images = s1ImageList.map(function(image, index) {
  var suffix = '_T' + (index + 1);
  var s1BandNames = ['VV', 'VH'];
  var newBandNames = s1BandNames.map(function(band) {
    return band + suffix;
  });
  return image.rename(newBandNames);
});

// Add to map for visualization
var s1VisParams = { bands: ['VV_T1'], min: 0, max: 1 };
Map.addLayer(ee.Image.cat(renamedS1Images), s1VisParams, 'Sentinel-1 Stack');

//=============================================================================
// 4. DIGITAL ELEVATION MODEL AND SLOPE
//=============================================================================

// Load Copernicus GLO-30 DEM
var glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30');

var glo30Filtered = glo30
  .filter(ee.Filter.bounds(geometry2))
  .select('DEM');

// Extract the projection
var demProj = glo30Filtered.first().select(0).projection();

// Create a mosaic and set the projection
var elevation = glo30Filtered.mosaic().rename('dem')
  .setDefaultProjection(demProj);

// Compute the slope
var slope = ee.Terrain.slope(elevation).rename('slope');

//=============================================================================
// 5. CANOPY HEIGHT
//=============================================================================

var canopyHeight = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
  .select('b1')
  .rename('CH');

//=============================================================================
// 6. ALOS PALSAR SAR DATA
//=============================================================================

var palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')
  .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
  .select(['HH', 'HV'])
  .mosaic()
  .rename(['PAL_HH', 'PAL_HV']);

//=============================================================================
// 7. MERGE ALL DATA INTO FINAL STACK
//=============================================================================

var finalStack = ee.Image.cat([
  mergedStackS2,         // Sentinel-2 optical bands
  mergedStackL8,         // Landsat-8 optical bands
  ee.Image.cat(renamedS1Images),  // Sentinel-1 SAR bands
  palsar,                // PALSAR SAR bands
  elevation,             // DEM
  slope,                 // Slope
  canopyHeight           // Canopy Height
])
.clip(geometry2)
.toDouble();

// Print band names to console for verification
print("Final stack band names:", finalStack.bandNames());

//=============================================================================
// 8. EXPORT DATA
//=============================================================================

// Export the final stack to Google Drive
Export.image.toDrive({
  image: finalStack,
  description: 's1_s2_l8_palsar_ch_dem_betul_2020_clipped',
  folder: 'ee-exports',
  fileNamePrefix: 's1_s2_l8_palsar_ch_betul_2020_clipped',
  region: geometry2,
  scale: 40,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export the biomass reference data
Export.image.toDrive({
  image: image.select('b1').clip(geometry2),
  description: 'agb_betul_clipped',
  folder: 'ee-exports',
  fileNamePrefix: 'agb_betul_clipped',
  region: geometry2,
  scale: 40,
  maxPixels: 1e13,
  crs: 'EPSG:4326',
  fileFormat: 'GeoTIFF'
});