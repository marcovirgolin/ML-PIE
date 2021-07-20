$("#btn-reset").on("click",function() {
    $.post('app.php', {
            functionname: "resetall",
        },
        function (obj, textstatus) {
            // success
            if (!('error' in obj)) {
                //console.log(obj)
                window.location = window.location.protocol + "//" + window.location.host;    
            }
            // request successful but error happened
            else {
                console.log(obj.error)
            }
        }
    )
})