/* begin HTML objects */
var divModelsContainer = $(".model-container")
var txtBoxTrap = $("#txtbox-not-a-trap")
//var btnSubmit = $("#btn-submit")
var divHiddenInfo = $("#div-hidden-info")
var divLoadingImg = $("#div-loading-img")
var w = $("h4.mb-0").width()/5;
/* end HTML objects */

// Do not show again models that were evaluated before
alreadySeenModels = new Set()
var stopModelRetrieval = false;
var current = []
var chosen = -1;

$(window).resize(function () {
    w = $("h4.mb-0").width()/4;
});

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


/* create model feedback container */
function generateModelFeedbackForm(models) {
    current = []
    formula_latex = $("h4.mb-0");
    formula_uncer = $("span.unc");
    formula_size = $("span.formula-size");
    for (var i = 0; i < 2; i++) {
        // store in already seen as to not display twice to the user
        alreadySeenModels.add(models[i].str_repr)

        // fill-in the html
        let new_model = models[i].str_repr;
        new_model = new_model.replaceAll("+1e-06","");
        console.log(new_model,models[i].str_repr );
        formula_latex[i].innerHTML = "$$" + new_model + "$$";
        console.log(formula_latex[i].innerHTML)
        w = $("h4.mb-0").width()/5;
        fsize = Math.min(1.5, w / models[i].str_repr.length);
        $(formula_latex[i]).attr("style", "font-size: " + fsize + "em;");
        // formula_uncer[i].innerHTML = models[i].uncertainty;
        formula_size[i].innerHTML = models[i].n_components;
        current.push(models[i]);
    }
    MathJax.typeset();
}


function retrieveModels() {
    if (stopModelRetrieval)
        return;

    divHiddenInfo.empty()
    divLoadingImg.attr("hidden", false)
    divModelsContainer.attr("hidden", true)
    models = []

    $.get('app.php', {
            functionname: "getexposedinfo",
        },
        function (obj, textstatus) {
            // success
            if (!('error' in obj) && obj['result']) {
                models = obj['result'];
                unseenModels = new Array()
                var k = 1
                for (var i = 0; i < models.length; i++) {
                    if (alreadySeenModels.has(models[i].str_repr))
                        continue;
                    
                    for (var j = i+1; j < models.length; j++) {
                        if (models[i] == models[j])
                            continue;
                    }
                    models[i].id = k++
                    unseenModels.push(models[i])
                }
                //console.log(unseenModels)
                if (unseenModels.length < 2) {
               	    models = []
               	    unseenModels = []
                    divLoadingImg.attr("hidden", false)
                    divModelsContainer.attr("hidden", true)
                    console.log('no new models, trying again in a while')
                    setTimeout(function () {
                        retrieveModels()
                    }, 200);
                } else {
                    divModelsContainer.attr("hidden", false)
                    divLoadingImg.attr("hidden", true)
                    generateModelFeedbackForm(unseenModels)
                    // mock sending feedback immediately
                }
 
            } else {
                // request successful but error happened
                console.log('Something went wrong in model retrieval, re-attempting in a while')
                unseenModels = []
                models = []
                divLoadingImg.attr("hidden", false)
                divModelsContainer.attr("hidden", true)
                setTimeout(function () {
                    retrieveModels()
                }, 200);
            }
        }
    ).fail(function (error) {
        console.log('Something failed in model retrieval, re-attempting in a while')
        unseenModels = []
        models = []
        divLoadingImg.attr("hidden", false)
        divModelsContainer.attr("hidden", true)
        setTimeout(function () {
            retrieveModels()
        }, 200);
    });
}


var feedbackCounter = 0;
function sendFeedback() {
    humanFeedback = {}
    humanFeedback.id = feedbackCounter++;
    humanFeedback.logtime = new Date().getTime() / 1000
    humanFeedback.feedback = new Array()

    obj = {}
    obj["label"] = chosen
    
    other = (1 + chosen) % 2
    if (current[chosen].interpretability <  current[other].interpretability) {
        obj["misprediction"] = 1
    } else {
        obj["misprediction"] = 0
    }

    for (var j = 0; j < 2; j++) {
        key = "prefnot_repr" + (j + 1);
        obj[key] = current[j].prefnot_repr
        key = "predicted_interpr" + (j + 1);
        obj[key] = current[j].interpretability
        key = "uncert_interpr" + (j + 1);
        obj[key] = current[j].uncertainty
    }
    humanFeedback.feedback.push(obj);
    console.log('Sending human feedback:', humanFeedback)

    // save json of human feedback
    $.post('app.php', {
            functionname: "savehumanfeedback",
            humanfeedback: humanFeedback,
            problem: problemID,
        },

        function (obj, textstatus) {
            // success
            if (!('error' in obj)) {
                //console.log(obj.result)
            }
            // request successful but error happened
            else {
                console.log(obj.error)
            }
        }
    )
}

function runPyNSGP() {
    $.post('app.php', {
            functionname: "runpynsgp",
        },

        function (obj, textstatus) {
            // success
            if (!('error' in obj)) {
                //console.log(obj.result)
            }
            // request successful but error happened
            else {
                console.log(obj.error)
            }
        }
    )
}


/* begin start up */
retrieveModels()
/* end start up */

divModelsContainer.click(function () {
    divModelsContainer.each(function () {
        $(this).removeClass("selected");
    });
    $(this).addClass("selected");
    var i = $("input[type=hidden]", this).val();
    //console.log(i);
    //console.log(current[parseInt(i)]);
    chosen = parseInt(i);
    //btnSubmit.attr("disabled", false);
    sendFeedback();
    // clear everything up
    retrieveModels()
    divModelsContainer.each(function () {
        $(this).removeClass("selected");
    });
});

// Legend
var problemID = null;
$(function(){
    $.get('app.php', {
            functionname: "getlegend",
        },
        function (obj, textstatus) {
            if (!('error' in obj)) {
                legendHtml = obj['result'];
                $("#legend").html(legendHtml);
                problemID = $("#problem-id").text().toLowerCase();
                setTimeout(function () {
                    MathJax.typeset();
                }, 500);
            } else {
                console.log('Error in retrieve legend');
            }
        }
    ).fail(function(err){
        console.log('Failed to retrieve legend');
    });
});

// Progress bar
var maxGenerations = null;
var currGeneration = 0;
function doesFileExist(urlToFile) {
    var xhr = new XMLHttpRequest();
    xhr.open('HEAD', urlToFile, false);
    xhr.send();
    if (xhr.status == "404") {
        return false;
    } else {
        return true;
    }
}

function updateProgress() {
    // understand what problem it is
    problemID = $("#problem-id").text().toLowerCase();
    // check it was identified, else bye
    if (problemID == ""){
        console.log("progress bar: problemID not found")
        return;
    }
    // retrieving progress by looking at the log files

    // if maxGenerations is not set, attempt to set it
    if (maxGenerations == null){
        $.get('app.php', {
            functionname: "getmaxgenerations",
            problem: problemID,
            },
            function (obj, textstatus) {
                if (!('error' in obj)) {
                    maxGenerations = obj['result'];
                } else {
                    console.log('Error in retrieve max generations');
                }
            }
        ).fail(function(err){
            console.log('Failed to retrieve max generations');
        });
    }

    // try to update the progress
    if (maxGenerations != null){
        $.get('app.php', {
            functionname: "getcurrgeneration",
            problem: problemID,
        },
        function (obj, textstatus) {
            if (!('error' in obj)) {
                currGeneration = obj['result'];
                progressPercentage = currGeneration * 100 / maxGenerations
                progressBar = $("#evolution-progress-bar")
                progressBar.width(progressPercentage+"%")
                progressBar.attr("aria-valuenow", progressPercentage)
                progressBar.html(progressPercentage+"%")
            } else {
                console.log('Error in retrieve current generation');
            }
        }).fail(function(err){
            console.log('Failed to current generation');
        });
    }
}


$(function(){
    setInterval(function(){
        if (maxGenerations!=null && currGeneration == maxGenerations) {
            // we finished, load survey page
            stopModelRetrieval = true;
            $("#btn-proceed").attr("disabled", false);
            divLoadingImg.attr("hidden", true);
            divModelsContainer.attr("hidden", true);
        }
        // update progress (bar)
        updateProgress();
    }, 500);
})

$("#btn-proceed").on("click", function(){
    window.location = window.location.protocol + "//" + window.location.host + "/" + "survey.html";
})


