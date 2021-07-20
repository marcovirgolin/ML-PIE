<?php
  header('Content-Type: application/json');
  session_start();
  if (!isset($_SESSION['uid'])) {
    $_SESSION['uid'] = uniqid();

  }
  if (!file_exists("./sessions/".$_SESSION['uid'])) {
    mkdir("./sessions/".$_SESSION['uid'], 0700, true);
  }

  $aResult = array();

  if( !isset($_POST['functionname']) && !isset($_GET['functionname']) ) { $aResult['error'] = 'Invalid GET/POST request: no function name provided.'; }

  /* FUNCTION DEFINITIONS */
  function test() {
    if( !isset($_POST['arguments']) ) { $aResult['error'] = 'Invalid POST request: No function arguments.'; }

    if( !is_array($_POST['arguments']) || (count($_POST['arguments']) < 2) ) {
      $aResult['error'] = 'Invalid POST request: error in arguments.';
    }
    else {
      $aResult['result'] = floatval($_POST['arguments'][0]) + floatval($_POST['arguments'][1]);
    }
    return $aResult;
  }

  function rrmDir($dir) {
    if (is_dir($dir)) {
      $objects = scandir($dir);
      foreach ($objects as $object) {
        if ($object != "." && $object != "..") {
          if (filetype($dir."/".$object) == "dir") 
             rrmdir($dir."/".$object); 
          else unlink($dir."/".$object);
        }
      }
      reset($objects);
      rmdir($dir);
    }
  }

  function deleteFeedback() {
    $problem = $_POST['problem'];
    rrmDir("./sessions/".$_SESSION['uid']."/out/".$problem);
    $aResult = array();
    $aResult['result'] = 'success';  
    return $aResult;
  }

  function resetAll(){
    rrmDir("./sessions/".$_SESSION['uid']."/out");
    $aResult = array();
    $aResult['result'] = 'success';  
    return $aResult;
  }

  function determineWhichProblemToRun() {
    $aResult = array();
    $boston_done = is_dir("./sessions/".$_SESSION['uid']."/out/boston");
    $german_done = is_dir("./sessions/".$_SESSION['uid']."/out/german");
    if ((!$boston_done) && (!$german_done)) {
      if (rand(0, 1) < 1) {
        $aResult['result'] = 'boston';
      } else {
        $aResult['result'] = 'german';
      }
    }
    else if (!$boston_done) {
      $aResult['result'] = 'boston';
    }
    else if (!$german_done) {
      $aResult['result'] =  'german';
    } else {
      $aResult['result'] = 'none';  
    }
    return $aResult;
  }

  function startPyNSGP() {
    $problem = determineWhichProblemToRun()['result'];
    # set the right legend
    unlink("./sessions/".$_SESSION['uid']."/curr_legend.html");
    # create session dir if it wasn't there already
    mkdir("./sessions/".$_SESSION['uid']."/", 0700, true);
    copy('./legends/'.$problem.'.html', "./sessions/".$_SESSION['uid']."/curr_legend.html");
    # delete any possible pre-existing temp files
    unlink("./sessions/".$_SESSION['uid']."/exp_info.json");
    unlink("./sessions/".$_SESSION['uid']."/human_feedback.json");
    # run pynsgp in background (works only on unix)
    $cmd = "python ../run.py ".$problem." ".$_SESSION['uid']." > /dev/null 2>&1 &";
    #$cmd = escapeshellcmd($cmd);
    exec($cmd);  
    $aResult = array();
    $aResult['result'] = 'success';
    return $aResult;
  }  

  function loadExposedInfo() {
    $string = file_get_contents("./sessions/".$_SESSION['uid']."/exp_info.json");
    $json_a = json_decode($string, true);
    $aResult = array();
    $aResult['result'] = $json_a;
    return $aResult;
  }

  function loadSurveyModels() {
    $what_problem = $_GET['problem'];
    $string = file_get_contents("./sessions/".$_SESSION['uid']."/out/".$what_problem."/survey_models.json");
    $json_a = json_decode($string, true);
    $aResult = array();
    $aResult['result'] = $json_a;
    return $aResult;
  }

  function getLegend() {
    $string = file_get_contents("./sessions/".$_SESSION['uid']."/curr_legend.html");
    $aResult = array();
    $aResult['result'] = $string;
    return $aResult;
  }
  
  function saveHumanFeedback() {
    $aResult = array();
    $objToSave = $_POST['humanfeedback'];
    if (!empty($objToSave)) {
      $fp = fopen("./sessions/".$_SESSION['uid']."/human_feedback.json", 'w');
      fwrite($fp, json_encode($objToSave, JSON_PRETTY_PRINT));   // here it will print the array pretty
      fclose($fp);
      // file_put_contents('human_feedback.json', $objToSave);
      /*
      // also log human feedback
      $problemID = $_POST['problem'];
      echo 'out/'.$problemID.'/human_feedback_'.$objToSave['id'].'.json';
      $fp = fopen('./out/'.$problemID.'/human_feedback_'.$objToSave['id'].'.json', 'w');
      fwrite($fp, json_encode($objToSave, JSON_PRETTY_PRINT));   // here it will print the array pretty
      fclose($fp);
      */
    }
    
    $aResult['result'] = 'success';
    return $aResult;
  }

  function getUid() {
    $aResult = array();
    $aResult['result'] = $_SESSION['uid'];
    return $aResult;
  }

  function getMaxGenerations() {
    // get max generations
    $what_problem = $_GET['problem'];
    $string = file_get_contents("./sessions/".$_SESSION['uid']."/out/".$what_problem."/log_generation_1.json");
    $json_a = json_decode($string, true);
    $max_generations = $json_a["generations_to_go"];
    if ($max_generations != null){
      $max_generations = intval($json_a["generations_to_go"]) + 1;
    }
    $aResult = array();
    $aResult['result'] = $max_generations;
    return $aResult;
  }

  function getCurrGeneration(){
    $what_problem = $_GET['problem'];
    $current_generation = 1;
    $next_generation = $current_generation + 1;
    $log_file = "./sessions/".$_SESSION['uid']."/out/".$what_problem."/log_generation_".strval($next_generation).".json";
    $log_file_exists = is_file($log_file);
    while($log_file_exists){
      $current_generation = $next_generation;
      $next_generation = $current_generation + 1;
      $log_file = "./sessions/".$_SESSION['uid']."/out/".$what_problem."/log_generation_".strval($next_generation).".json";
      $log_file_exists = is_file($log_file);
    }
    $aResult = array();
    $aResult['result'] = $current_generation;
    return $aResult;
  }


  /* REQUEST HANDLING */
  if( !isset($aResult['error']) ) {
    // GET
    if (isset($_GET['functionname'])){
      switch($_GET['functionname']) {
        case 'getuid':
          $aResult = getUid();
          break;
        case 'getproblemtorun':
          $aResult = determineWhichProblemToRun();
          break;
        case 'getlegend':
          $aResult = getLegend();
          break;
        case 'getexposedinfo':
          $aResult = loadExposedInfo();
          break;
        case 'getsurveymodels':
          $aResult = loadSurveyModels();
          break;
        case 'getmaxgenerations':
          $aResult = getMaxGenerations();
          break;
        case 'getcurrgeneration':
          $aResult = getCurrGeneration();
          break;
        default:
          $aResult['error'] = 'Invalid GET request: function '.$_GET['functionname'].' does not exist.';
          break;
      }
    }

    // POST
    else if (isset($_POST['functionname'])){
      switch($_POST['functionname']) {
        case 'test':
          $aResult = test();
          break;
        case 'deletefeedback':
          $aResult = deleteFeedback();
          break;
        case 'resetall':
          $aResult = resetAll();
          break;
        case 'runpynsgp':
          $aResult = startPyNSGP();
          break;
        case 'savehumanfeedback':
          $aResult = saveHumanFeedback();
          break;
        default:
          $aResult['error'] = 'Invalid POST request: function '.$_POST['functionname'].' does not exist.';
          break;
      }
    }
  }
  echo json_encode($aResult);
?>