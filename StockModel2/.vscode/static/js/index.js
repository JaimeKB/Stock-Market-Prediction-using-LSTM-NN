var obj_csv = {
    size:0,
    dataFile:[]
};

function readImage(input) {
  if (input.files && input.files[0]) {               

    let reader = new FileReader();
    reader.readAsBinaryString(input.files[0]);
    reader.onload = function (e) {
      obj_csv.size = e.total;
      obj_csv.dataFile = e.target.result
      parseData(obj_csv.dataFile)          
    }
  }
}

function parseData(data){
  let csvData = [];
  let lbreak = data.split("\n");
  lbreak.forEach(res => {
 
    if((res.split(",")).length == 7) {
      csvData.push(res.split(","));
    }
    
  });
  localStorage.setItem("fileData", csvData);

}

function loadChart(xDatelabels, yStockData) {

let ctx = document.getElementById('myChart').getContext('2d');
  let myChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: xDatelabels,
          datasets: [{
              label: 'Stock price predictions',
              data: yStockData,
              fill: false,
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              borderColor: 'rgba(255, 99, 132, 1)',
              borderWidth: 1
          }]
      },
  });
}

function load2Charts(xDatelabels, yStockData1, yStockData2) {
let ctx = document.getElementById('myChart').getContext('2d');
  let myChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: xDatelabels,
          datasets: [
            {
              label: "User's model Stock price predictions",
              data: yStockData1,
              fill: false,
              backgroundColor: 'rgba(222, 17, 2, 0.2)',
              borderColor: 'rgba(222, 17, 2, 1)',
              borderWidth: 1
            },
            {
              label: "Software's model Stock price predictions",
              data: yStockData2,
              fill: false,
              backgroundColor: 'rgba(45, 5, 227, 0.2)',
              borderColor: 'rgba(45, 5, 227, 1)',
              borderWidth: 1
            }
        ]
      },
  });
}

$(document).ready(function(){

  $('#upload-CSV-file-btn').click(function() {
    var form_data = new FormData($('#upload-csv')[0]);
    $.ajax({
        type: 'POST',
        url: '/uploadCSV',
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        success: function(dataDict) {
            message = dataDict['message']
            predictedData = dataDict['predictedData']
            testData = dataDict['testData']
            dateRange = dataDict['dateRange']
            dataLength = dataDict['dataLength']                      

            chartData = []

            for (i = 0; i < predictedData.length; i++) {
              chartData.push(predictedData[i][0])
            }

            loadChart(dateRange.slice(0, 90), chartData.slice(0, 90))
        },
    });

  });

  $('#upload-h5-file-btn').click(function() {
    var form_data = new FormData($('#upload-file')[0]);
    $.ajax({
        type: 'POST',
        url: '/uploadajax',
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        success: function(dataDict) {
            console.log('Success!');

            predictedData = dataDict['predictedData']
            myModelPredictedData = dataDict['myModelPredictedData']
            dateRange = dataDict['dateRange']

            chartData1 = []
            chartData2 = []

            for (i = 0; i < predictedData.length; i++) {
              chartData1.push(predictedData[i][0])
              chartData2.push(myModelPredictedData[i][0])
            }

            load2Charts(dateRange.slice(0, 150), chartData1.slice(0, 150), chartData2.slice(0, 150))
        },
    });

  });
});