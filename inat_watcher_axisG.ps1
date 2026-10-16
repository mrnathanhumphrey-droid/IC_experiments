$ErrorActionPreference = "Continue"
$ip   = "129.80.195.93"
$id   = "654e60943e47455fa54255b2c2649767"
$key  = "D:\IC_experiments\.ssh\lambda_ic"
$res  = "D:\IC_experiments\tier_abstraction_v0_1\results"
$opts = @("-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=NUL","-o","LogLevel=ERROR","-o","ConnectTimeout=20")
$apikey = [Environment]::GetEnvironmentVariable("LAMBDA_API_KEY","User")
$b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("$apikey`:"))
$h = @{ Authorization = "Basic $b64" }
$capMin = 60; $staleMin = 20; $t0 = Get-Date

function Term-Verify {
  $body = '{"instance_ids":["' + $id + '"]}'
  try { Invoke-RestMethod -Uri "https://cloud.lambda.ai/api/v1/instance-operations/terminate" -Headers $h -Method POST -ContentType "application/json" -Body $body | Out-Null } catch { "term err: $($_.Exception.Message)" }
  Start-Sleep -Seconds 8
  try { $r = Invoke-RestMethod -Uri "https://cloud.lambda.ai/api/v1/instances" -Headers $h -Method GET
        "VERIFY: alive=$(@($r.data | Where-Object { $_.id -eq $id }).Count) (0=good)" } catch { "verify err: $($_.Exception.Message)" }
}
function Pull-Log { & scp @opts -i $key "ubuntu@${ip}:dt/run.log" "$res\axisG_run.log" 2>$null }

while ($true) {
  $elapsed = [int]((Get-Date) - $t0).TotalMinutes
  $probe = & ssh @opts -i $key ubuntu@$ip "test -f /lambda/nfs/iNat/granularity_axisG_supersession.json && echo DONE; echo AGE=`$(( `$(date +%s) - `$(stat -c %Y dt/run.log 2>/dev/null || date +%s) )); grep -aq Traceback dt/run.log 2>/dev/null && echo CRASH" 2>$null
  $probe = ($probe -join "`n")
  if ([string]::IsNullOrWhiteSpace($probe)) { "t=${elapsed}m :: SSH blank -> hold"; Start-Sleep -Seconds 60; continue }
  $done = $probe -match "DONE"; $crash = $probe -match "CRASH"
  $age = if ($probe -match "AGE=(\d+)") { [int]$Matches[1] } else { 0 }
  "t=${elapsed}m :: done=$done crash=$crash logage=$([int]($age/60))m"
  if ($done)  { "RESULT FOUND -> pulling"; & scp @opts -i $key "ubuntu@${ip}:/lambda/nfs/iNat/granularity_axisG_supersession.json" "$res\granularity_axisG_supersession.json" 2>$null; Pull-Log; Term-Verify; "WATCHER DONE (success)"; break }
  if ($crash) { "TRACEBACK -> pulling, terminating"; Pull-Log; Term-Verify; "WATCHER DONE (crash)"; break }
  if ($age -ge ($staleMin*60)) { "LOG STALE ${staleMin}m -> terminating"; Pull-Log; Term-Verify; "WATCHER DONE (stall)"; break }
  if ($elapsed -ge $capMin) { "CAP $capMin m -> terminating"; Pull-Log; Term-Verify; "WATCHER DONE (cap)"; break }
  Start-Sleep -Seconds 60
}
