<?php
    header('Content-Type: application/json');
    session_start();

    if (!isset($_SESSION['uid'])) {
        trigger_error("UID NOT SET WTF", E_ERROR);
    }

    $data = json_decode(file_get_contents('php://input'), true);

    $problemPathsLookup = array(
        'boston' => 'boston',
        'german' => 'german'
    );

    $path = "./sessions/".$_SESSION['uid']."/out/{$problemPathsLookup[$data["problem"]]}/survey.json";
    $fp = fopen($path, 'w');
    fwrite($fp, json_encode($data));
    fclose($fp);

    $zip = new ZipArchive();
    $zpath = "./sessions/".$_SESSION['uid']."/out/{$problemPathsLookup[$data["problem"]]}.zip";
    $zip->open($zpath, ZipArchive::CREATE | ZipArchive::OVERWRITE);

    $rootPath = "./sessions/".$_SESSION['uid']."/out/{$problemPathsLookup[$data["problem"]]}/";
    $files = new RecursiveIteratorIterator(
    new RecursiveDirectoryIterator($rootPath),
        RecursiveIteratorIterator::LEAVES_ONLY
    );

    foreach ($files as $name => $file){
        if (!$file->isDir()){
            $filePath = $file->getRealPath();
            $relativePath = substr($filePath, strlen($rootPath) + 1);

            $zip->addFile($filePath, $relativePath);
        }
    }

    $zip->close();

    $curlFile = curl_file_create($zpath);
    $post = array(
        'file_contents'=> $curlFile,
        'file_name' => $problemPathsLookup[$data["problem"]],
        'uid' => $_SESSION['uid']
        );
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL,"http://SERVER WHERE YOU WANT TO STORE THIS/feedback");
    curl_setopt($ch, CURLOPT_POST,1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $post);
    $result=curl_exec ($ch);
    curl_close ($ch);
?>