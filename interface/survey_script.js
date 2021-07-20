// Legend
var problemID = null
var problemToRun = null;
var types = ["online", "phi", "size"]

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

function getProblemToRun() {
    $.get('app.php', {
            functionname: "getproblemtorun",
        },

        function (obj, textstatus) {
            // success
            if (!('error' in obj)) {
                problemToRun = obj.result;
                if (problemToRun == 'boston')
                    $('#put-submit-btn-text-here').text("Proceed with Boston Housing");
                else if (problemToRun == 'german')
                    $('#put-submit-btn-text-here').text("Proceed with German Credit");
                else
                    $('#put-submit-btn-text-here').text("Submit & conclude");            
            }
            // request successful but error happened
            else {
                console.log(obj.error)
            }
        }
    )
}




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


$(function () {

    // set next problem to run & btn-submit behavior
    getProblemToRun();

    // added timeout here: maybe survey_models.json is not ready yet!
    setTimeout(function(){
        // Legend
        $(function(){
            $.get('app.php', {
                    functionname: "getlegend",
                },
                function (obj, textstatus) {
                    if (!('error' in obj)) {
                        legendHtml = obj['result'];
                        $("#legend").html(legendHtml);
                        problemID = $("#problem-id").text().toLowerCase();
                        $.get('app.php', {
                                functionname: "getsurveymodels",
                                problem: problemID,
                            },
                            function (obj, textstatus) {
                                if (!('error' in obj)) {
                                    models = obj['result'];
                                    $("#div-loading-img").hide();
                                    for (let i = 0; i < 2; i++) {
                                        for (let j = 0; j < types.length; j++) {
                                            let t = types[j];
                                            $(".model" + i + "_" + t).each(function (u, e) {
                                                let new_model = models[i][t];
                                                new_model = new_model.replaceAll("+1e-06","");
                                                $(e).html("$$" + new_model + "$$");
                                                w = $("h4.mb-0").width()/5;
                                                fsize = Math.min(1.4, w / models[i][t].length);
                                                $(e).attr("style", "font-size: " + fsize + "em;");
                                            });
                                        }
                                    }
                                    MathJax.typeset();
                                } else {
                                    console.log('Error in retrieve survey models');
                                }
                            }
                        ).fail(function(err){
                            console.log('Failed to retrieve survey models');
                        });
                    } else {
                        console.log('Error in retrieve legend');
                    }
                }
            ).fail(function(err){
                console.log('Failed to retrieve legend');
            });
        });

    }, 1000);


    $("#btn-undo").click(function(e) {
        e.preventDefault();
        // delete feedback of that run
        $.post('app.php', {
                functionname: "deletefeedback",
                problem: problemID,
            },
            function (obj, textstatus) {
                // success
                if (!('error' in obj)) {
                    //redirect to main page
                     window.location = window.location.protocol + "//" + window.location.host;  
                }
                // request successful but error happened
                else {
                    console.log(obj.error)
                }
            }
        );
    });

    $("#btn-submit").click(function (e) {
        // send us the material
        e.preventDefault();        

        let radio_groups = {}
        $(":radio").each(function () {
            if ($(this).is(':checked')) radio_groups[this.id] = true;
        });
        let feedback = {
            problem: problemID,
            results: radio_groups
        }
        $.ajax({
            url: 'survey.php',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(feedback),
            dataType: 'json'
        });

        // run next problem or terminate
        if (problemToRun != "none") {
            runPyNSGP(function(){
                window.location = window.location.protocol + "//" + window.location.host + "/" + "feedback.html";    
            });
        } else {
            window.location = window.location.protocol + "//" + window.location.host + "/" + "thanks.html";  
        }
    });

});
