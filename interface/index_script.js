function runPyNSGP(successFun) {
    $.post('app.php', {
            functionname: "runpynsgp",
        },

        function (obj, textstatus) {
            // success
            if (!('error' in obj)) {
                //console.log(obj.result)
                successFun()
            }
            // request successful but error happened
            else {
                console.log(obj.error)
            }
        }
    )
}

function getUid() {
    $.get('app.php', {
            functionname: "getuid",
        },
        function (obj, textstatus) {
            if (!('error' in obj)) {
                uid = obj['result'];
                console.log(uid);
            } else {
                console.log('Error in retrieve UID');
            }
        }
    ).fail(function(err){
        console.log('Failed to retrieve UID');
    });
}


var problemToRun = null;
function getProblemToRun() {
    $.get('app.php', {
            functionname: "getproblemtorun",
        },

        function (obj, textstatus) {
            // success
            if (!('error' in obj)) {
                problemToRun = obj.result;
                if (problemToRun == 'boston')
                    $('#put-problem-name-here').text("Boston Housing");
                else if (problemToRun == 'german')
                    $('#put-problem-name-here').text("German Credit");
                else {
                    // they already did it, redirect to thanks.html
                    window.location = window.location.protocol + "//" + window.location.host + "/" + "thanks.html";    
                }
            }
            // request successful but error happened
            else {
                console.log(obj.error)
            }
        }
    )
}

$("#btn-proceed").on("click",function() {
    runPyNSGP(function(){
        window.location = window.location.protocol + "//" + window.location.host + "/" + "feedback.html";    
    });
})

// initialize
$(function(){
    getProblemToRun()
});