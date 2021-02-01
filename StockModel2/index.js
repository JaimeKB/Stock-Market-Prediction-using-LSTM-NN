$(document).ready(function(){
    
    $("uploadData").click(function() {
        console.log("Done")
        $.ajax({
            
            // url:"/FileData/"+$("#stockData").value,
            // success: function(){
            //     console.log("File uploaded")
            //     // $("#MainTable").append(table);
            // }
        });        
    });

});