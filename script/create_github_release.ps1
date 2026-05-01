#!/usr/bin/env powershell

[CmdletBinding()]
param(
    [ValidateSet('dev', 'preview', 'stable', 'oss')]
    [string]$Channel = 'oss',

    [Parameter(Mandatory = $true)]
    [string]$AssetPath,

    [string]$Tag = '',
    [string]$BaseRef = '',
    [string]$ToRef = 'HEAD',
    [string]$Repo = '',
    [string]$AuthorPattern = 'cesar|cesaryuan|cesaryuan@qq\.com',
    [string]$NotesPath = '',

    [switch]$Draft = $false,
    [switch]$Prerelease = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Assert-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command -Name $Name -Type Application -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Invoke-ExternalCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $rendered = ($output | ForEach-Object { "$_" }) -join "`n"
        throw "Command failed: $FilePath $($Arguments -join ' ')`n$rendered"
    }

    return (($output | ForEach-Object { "$_" }) -join "`n").Trim()
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

function Test-RemoteTagExists {
    param([Parameter(Mandatory = $true)][string]$ReleaseTag)

    $output = Invoke-ExternalCapture -FilePath 'git' -Arguments @(
        'ls-remote', '--tags', 'origin', "refs/tags/$ReleaseTag"
    )
    return -not [string]::IsNullOrWhiteSpace($output)
}

function Test-LocalTagExists {
    param([Parameter(Mandatory = $true)][string]$ReleaseTag)

    & git rev-parse --verify --quiet "refs/tags/$ReleaseTag" *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-GitHubRepo {
    param([string]$ExplicitRepo)

    if (-not [string]::IsNullOrWhiteSpace($ExplicitRepo)) {
        return $ExplicitRepo
    }

    $originUrl = Invoke-ExternalCapture -FilePath 'git' -Arguments @('remote', 'get-url', 'origin')

    if ($originUrl -match '^git@github\.com:(?<repo>.+?)(\.git)?$') {
        return $matches.repo
    }

    if ($originUrl -match '^https://github\.com/(?<repo>.+?)(\.git)?$') {
        return $matches.repo
    }

    throw "Could not infer GitHub repo from origin URL: $originUrl"
}

function Get-ChannelConfig {
    param([Parameter(Mandatory = $true)][string]$ReleaseChannel)

    $configPath = Join-Path $RepoRoot '.github/workflows/release_configurations.json'
    if (-not (Test-Path -LiteralPath $configPath)) {
        return $null
    }

    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    return $config.channels | Where-Object { $_.channel -eq $ReleaseChannel } | Select-Object -First 1
}

function Get-CommitSha {
    param([Parameter(Mandatory = $true)][string]$Ref)

    return (Invoke-ExternalCapture -FilePath 'git' -Arguments @('rev-parse', "$Ref^{commit}")).Trim()
}

function New-ReleaseTag {
    param([Parameter(Mandatory = $true)][string]$ReleaseChannel)

    $dateFormatted = Get-Date -Format 'yyyy.MM.dd.HH.mm'
    $baseTag = "v0.$dateFormatted.$ReleaseChannel"
    $suffix = 0

    while ($true) {
        $candidate = '{0}_{1:d2}' -f $baseTag, $suffix
        if (-not (Test-LocalTagExists -ReleaseTag $candidate) -and -not (Test-RemoteTagExists -ReleaseTag $candidate)) {
            return $candidate
        }

        $suffix += 1
    }
}

function Resolve-BaseRef {
    param(
        [string]$ExplicitBaseRef,
        [string]$TargetTag,
        [string]$TargetRef
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitBaseRef)) {
        return $ExplicitBaseRef
    }

    $tagOutput = Invoke-ExternalCapture -FilePath 'git' -Arguments @(
        'for-each-ref',
        'refs/tags',
        '--merged',
        $TargetRef,
        '--sort=-creatordate',
        '--format=%(refname:short)'
    )

    $tags = @()
    if (-not [string]::IsNullOrWhiteSpace($tagOutput)) {
        $tags = $tagOutput -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    }

    foreach ($candidate in $tags) {
        if ($candidate -ne $TargetTag) {
            return $candidate
        }
    }

    return ''
}

function Get-CommitSubjects {
    param(
        [string]$StartRef,
        [Parameter(Mandatory = $true)][string]$EndRef,
        [Parameter(Mandatory = $true)][string]$CommitAuthorPattern
    )

    $arguments = @('log', '--no-merges', '--format=%an%x09%ae%x09%s')
    if (-not [string]::IsNullOrWhiteSpace($StartRef)) {
        $arguments += "$StartRef..$EndRef"
    } else {
        $arguments += $EndRef
    }

    $output = Invoke-ExternalCapture -FilePath 'git' -Arguments $arguments
    if ([string]::IsNullOrWhiteSpace($output)) {
        return @()
    }

    $subjects = New-Object System.Collections.Generic.List[string]
    foreach ($line in ($output -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        $parts = $line -split "`t", 3
        if ($parts.Count -lt 3) {
            continue
        }

        $authorName = $parts[0]
        $authorEmail = $parts[1]
        $subject = $parts[2]
        if ($subject -match '^Merge\b') {
            continue
        }

        if ($authorName -match $CommitAuthorPattern -or $authorEmail -match $CommitAuthorPattern) {
            $subjects.Add($subject)
        }
    }

    return $subjects
}

function Build-ReleaseNotes {
    param(
        [string[]]$Subjects,
        [string]$StartRef,
        [string]$EndRef
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('# Release Notes')

    if (-not [string]::IsNullOrWhiteSpace($StartRef)) {
        $lines.Add(("Commit range: `{0}..{1}" -f $StartRef, $EndRef))
    } else {
        $lines.Add(("Commit ref: `{0}" -f $EndRef))
    }

    $lines.Add('')

    if ($Subjects.Count -eq 0) {
        $lines.Add('No matching commit messages found.')
    } else {
        foreach ($subject in $Subjects) {
            $lines.Add("- $subject")
        }
    }

    return ($lines -join "`n").Trim() + "`n"
}

function Get-NotesFilePath {
    param(
        [string]$ExplicitPath,
        [Parameter(Mandatory = $true)][string]$ReleaseTag
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $directory = Split-Path -Parent $ExplicitPath
        if (-not [string]::IsNullOrWhiteSpace($directory)) {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        }
        return (Resolve-Path -LiteralPath (New-Item -ItemType File -Force -Path $ExplicitPath).FullName).Path
    }

    $notesDirectory = Join-Path $RepoRoot 'target/release-notes'
    New-Item -ItemType Directory -Force -Path $notesDirectory | Out-Null
    return (Join-Path $notesDirectory "$ReleaseTag.md")
}

function Test-ReleaseExists {
    param(
        [Parameter(Mandatory = $true)][string]$ReleaseTag,
        [Parameter(Mandatory = $true)][string]$GitHubRepo
    )

    & gh release view $ReleaseTag --repo $GitHubRepo --json tagName *> $null
    return $LASTEXITCODE -eq 0
}

function Find-ExistingReleaseTagForCommit {
    param(
        [Parameter(Mandatory = $true)][string]$CommitSha,
        [Parameter(Mandatory = $true)][string]$ReleaseChannel,
        [Parameter(Mandatory = $true)][string]$GitHubRepo
    )

    $tagOutput = Invoke-ExternalCapture -FilePath 'git' -Arguments @(
        'tag', '--points-at', $CommitSha, '--sort=-version:refname'
    )
    if ([string]::IsNullOrWhiteSpace($tagOutput)) {
        return ''
    }

    $channelPattern = '\.{0}_[0-9][0-9]$' -f [System.Text.RegularExpressions.Regex]::Escape($ReleaseChannel)
    foreach ($candidate in ($tagOutput -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }

        if ($candidate -notmatch $channelPattern) {
            continue
        }

        if (Test-ReleaseExists -ReleaseTag $candidate -GitHubRepo $GitHubRepo) {
            return $candidate
        }
    }

    return ''
}

function Get-ReleaseBody {
    param(
        [Parameter(Mandatory = $true)][string]$ReleaseTag,
        [Parameter(Mandatory = $true)][string]$GitHubRepo
    )

    return Invoke-ExternalCapture -FilePath 'gh' -Arguments @(
        'release', 'view', $ReleaseTag, '--repo', $GitHubRepo, '--json', 'body', '--jq', '.body // ""'
    )
}

function Test-ShouldPreserveExistingNotes {
    param(
        [string]$ExistingBody,
        [string]$GeneratedBody,
        [string]$DefaultBody
    )

    if ([string]::IsNullOrWhiteSpace($ExistingBody)) {
        return $false
    }

    $normalizedExisting = $ExistingBody.Trim()
    $normalizedGenerated = if ($null -eq $GeneratedBody) { '' } else { $GeneratedBody.Trim() }
    $normalizedDefault = if ($null -eq $DefaultBody) { '' } else { $DefaultBody.Trim() }

    if ($normalizedExisting -eq $normalizedGenerated) {
        return $false
    }

    if (-not [string]::IsNullOrWhiteSpace($normalizedDefault) -and $normalizedExisting -eq $normalizedDefault) {
        return $false
    }

    return $true
}

Assert-CommandExists -Name 'git'

$resolvedAssetPath = [System.IO.Path]::GetFullPath($AssetPath)
if (-not (Test-Path -LiteralPath $resolvedAssetPath -PathType Leaf)) {
    throw "Asset not found: $resolvedAssetPath"
}

$gitHubRepo = Resolve-GitHubRepo -ExplicitRepo $Repo
$channelConfig = Get-ChannelConfig -ReleaseChannel $Channel
$releaseBaseName = if ($null -ne $channelConfig) { $channelConfig.release_base_name } else { "$Channel Release" }
$defaultReleaseBody = if ($null -ne $channelConfig) { $channelConfig.release_body_text } else { '' }

$reusedExistingRelease = $false
$targetCommitSha = Get-CommitSha -Ref $ToRef

if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = New-ReleaseTag -ReleaseChannel $Channel
}

$effectiveBaseRef = Resolve-BaseRef -ExplicitBaseRef $BaseRef -TargetTag $Tag -TargetRef $ToRef
$commitSubjects = Get-CommitSubjects -StartRef $effectiveBaseRef -EndRef $ToRef -CommitAuthorPattern $AuthorPattern
$notesBody = Build-ReleaseNotes -Subjects $commitSubjects -StartRef $effectiveBaseRef -EndRef $ToRef
$resolvedNotesPath = Get-NotesFilePath -ExplicitPath $NotesPath -ReleaseTag $Tag
Set-Content -LiteralPath $resolvedNotesPath -Value $notesBody -Encoding utf8

$title = "$releaseBaseName $Tag"

Write-Output "Repo: $gitHubRepo"
Write-Output "Tag: $Tag"
Write-Output "Title: $title"
Write-Output "Asset: $resolvedAssetPath"
Write-Output "BaseRef: $effectiveBaseRef"
Write-Output "ToRef: $ToRef"
Write-Output "TargetCommit: $targetCommitSha"
Write-Output "ReusedExistingRelease: $reusedExistingRelease"
Write-Output "Notes: $resolvedNotesPath"

if ($DryRun) {
    Write-Output ''
    Write-Output '--- Release Notes Preview ---'
    Write-Output $notesBody
    exit 0
}

Assert-CommandExists -Name 'gh'

Invoke-External -FilePath 'gh' -Arguments @('auth', 'status')
Invoke-External -FilePath 'git' -Arguments @('fetch', '--tags', 'origin')

$existingReleaseTag = Find-ExistingReleaseTagForCommit -CommitSha $targetCommitSha -ReleaseChannel $Channel -GitHubRepo $gitHubRepo
if (-not [string]::IsNullOrWhiteSpace($existingReleaseTag) -and $existingReleaseTag -ne $Tag) {
    $Tag = $existingReleaseTag
    $reusedExistingRelease = $true
    $effectiveBaseRef = Resolve-BaseRef -ExplicitBaseRef $BaseRef -TargetTag $Tag -TargetRef $ToRef
    $commitSubjects = Get-CommitSubjects -StartRef $effectiveBaseRef -EndRef $ToRef -CommitAuthorPattern $AuthorPattern
    $notesBody = Build-ReleaseNotes -Subjects $commitSubjects -StartRef $effectiveBaseRef -EndRef $ToRef
    $resolvedNotesPath = Get-NotesFilePath -ExplicitPath $NotesPath -ReleaseTag $Tag
    Set-Content -LiteralPath $resolvedNotesPath -Value $notesBody -Encoding utf8
    $title = "$releaseBaseName $Tag"
}

Write-Output "Using release tag: $Tag"
Write-Output "Reused existing release: $reusedExistingRelease"

if (-not (Test-LocalTagExists -ReleaseTag $Tag) -and -not (Test-RemoteTagExists -ReleaseTag $Tag)) {
    Invoke-External -FilePath 'git' -Arguments @('tag', $Tag, $ToRef)
    Invoke-External -FilePath 'git' -Arguments @('push', 'origin', $Tag)
} elseif ((Test-LocalTagExists -ReleaseTag $Tag) -and -not (Test-RemoteTagExists -ReleaseTag $Tag)) {
    Invoke-External -FilePath 'git' -Arguments @('push', 'origin', $Tag)
}

if (Test-ReleaseExists -ReleaseTag $Tag -GitHubRepo $gitHubRepo) {
    $existingBody = Get-ReleaseBody -ReleaseTag $Tag -GitHubRepo $gitHubRepo
    $existingReleasePointsToTarget = (Get-CommitSha -Ref $Tag) -eq $targetCommitSha
    $preserveExistingNotes = Test-ShouldPreserveExistingNotes -ExistingBody $existingBody -GeneratedBody $notesBody -DefaultBody $defaultReleaseBody

    if (-not $existingReleasePointsToTarget -or -not $preserveExistingNotes) {
        $editArguments = @(
            'release', 'edit', $Tag,
            '--repo', $gitHubRepo,
            '--title', $title
        )

        if (-not $preserveExistingNotes) {
            $editArguments += @('--notes-file', $resolvedNotesPath)
        }

        if ($Draft) {
            $editArguments += '--draft'
        }
        if ($Prerelease) {
            $editArguments += '--prerelease'
        }
        Invoke-External -FilePath 'gh' -Arguments $editArguments
    }

    Invoke-External -FilePath 'gh' -Arguments @('release', 'upload', $Tag, $resolvedAssetPath, '--repo', $gitHubRepo, '--clobber')
} else {
    $createArguments = @(
        'release', 'create', $Tag, $resolvedAssetPath,
        '--repo', $gitHubRepo,
        '--title', $title,
        '--notes-file', $resolvedNotesPath
    )
    if ($Draft) {
        $createArguments += '--draft'
    }
    if ($Prerelease) {
        $createArguments += '--prerelease'
    }
    Invoke-External -FilePath 'gh' -Arguments $createArguments
}

Write-Output ''
Write-Output "GitHub release ready: $title"
