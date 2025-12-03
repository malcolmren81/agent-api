-- CreateTable
CREATE TABLE "Session" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "isOnline" BOOLEAN NOT NULL DEFAULT false,
    "scope" TEXT,
    "expires" TIMESTAMP(3),
    "accessToken" TEXT NOT NULL,
    "userId" BIGINT,
    "firstName" TEXT,
    "lastName" TEXT,
    "email" TEXT,
    "accountOwner" BOOLEAN NOT NULL DEFAULT false,
    "locale" TEXT,
    "collaborator" BOOLEAN DEFAULT false,
    "emailVerified" BOOLEAN DEFAULT false,

    CONSTRAINT "Session_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Asset" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "imageUrl" TEXT,
    "thumbnailUrl" TEXT,
    "prompt" TEXT NOT NULL,
    "style" TEXT NOT NULL DEFAULT 'realistic',
    "dimensions" TEXT NOT NULL DEFAULT '1024x1024',
    "model" TEXT NOT NULL DEFAULT 'gemini',
    "status" TEXT NOT NULL DEFAULT 'processing',
    "cost" INTEGER NOT NULL DEFAULT 10,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "deletedAt" TIMESTAMP(3),

    CONSTRAINT "Asset_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Template" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "promptText" TEXT NOT NULL,
    "style" TEXT NOT NULL DEFAULT 'realistic',
    "tags" TEXT[],
    "language" TEXT NOT NULL DEFAULT 'en',
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdBy" TEXT,
    "usageCount" INTEGER NOT NULL DEFAULT 0,
    "acceptRate" DOUBLE PRECISION,
    "avgScore" DOUBLE PRECISION,
    "lastUsed" TIMESTAMP(3),
    "source" TEXT NOT NULL DEFAULT 'manual',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Template_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AgentLog" (
    "id" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "agentName" TEXT NOT NULL,
    "input" JSONB NOT NULL,
    "output" JSONB NOT NULL,
    "reasoning" TEXT,
    "executionTime" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "routingMode" TEXT,
    "usedLlm" BOOLEAN NOT NULL DEFAULT false,
    "confidence" DOUBLE PRECISION,
    "fallbackUsed" BOOLEAN NOT NULL DEFAULT false,
    "creditsUsed" INTEGER NOT NULL DEFAULT 0,
    "llmTokens" INTEGER,
    "modelName" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AgentLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ModelConfig" (
    "id" TEXT NOT NULL,
    "modelName" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "costPerGen" INTEGER NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "capabilities" JSONB NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ModelConfig_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ModelStats" (
    "id" TEXT NOT NULL,
    "modelName" TEXT NOT NULL,
    "bucket" TEXT NOT NULL,
    "impressions" INTEGER NOT NULL DEFAULT 0,
    "rewardMean" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "rewardVar" DOUBLE PRECISION NOT NULL DEFAULT 0.1,
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ModelStats_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Task" (
    "id" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "shop" TEXT NOT NULL,
    "originalPrompt" TEXT NOT NULL,
    "userRequest" JSONB,
    "stages" JSONB NOT NULL,
    "promptJourney" JSONB NOT NULL,
    "totalDuration" INTEGER NOT NULL,
    "creditsCost" INTEGER NOT NULL,
    "performanceBreakdown" JSONB NOT NULL,
    "evaluationResults" JSONB,
    "generatedImageUrl" TEXT,
    "mockupUrls" JSONB,
    "finalPrompt" TEXT,
    "status" TEXT NOT NULL,
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Task_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Asset_taskId_key" ON "Asset"("taskId");

-- CreateIndex
CREATE INDEX "Asset_shop_idx" ON "Asset"("shop");

-- CreateIndex
CREATE INDEX "Asset_status_idx" ON "Asset"("status");

-- CreateIndex
CREATE INDEX "Asset_createdAt_idx" ON "Asset"("createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "Template_name_key" ON "Template"("name");

-- CreateIndex
CREATE INDEX "Template_category_idx" ON "Template"("category");

-- CreateIndex
CREATE INDEX "Template_isActive_idx" ON "Template"("isActive");

-- CreateIndex
CREATE INDEX "Template_usageCount_idx" ON "Template"("usageCount");

-- CreateIndex
CREATE INDEX "AgentLog_taskId_idx" ON "AgentLog"("taskId");

-- CreateIndex
CREATE INDEX "AgentLog_shop_idx" ON "AgentLog"("shop");

-- CreateIndex
CREATE INDEX "AgentLog_agentName_idx" ON "AgentLog"("agentName");

-- CreateIndex
CREATE INDEX "AgentLog_routingMode_idx" ON "AgentLog"("routingMode");

-- CreateIndex
CREATE INDEX "AgentLog_createdAt_idx" ON "AgentLog"("createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "ModelConfig_modelName_key" ON "ModelConfig"("modelName");

-- CreateIndex
CREATE INDEX "ModelStats_bucket_idx" ON "ModelStats"("bucket");

-- CreateIndex
CREATE INDEX "ModelStats_modelName_idx" ON "ModelStats"("modelName");

-- CreateIndex
CREATE INDEX "ModelStats_lastUpdated_idx" ON "ModelStats"("lastUpdated");

-- CreateIndex
CREATE UNIQUE INDEX "ModelStats_modelName_bucket_key" ON "ModelStats"("modelName", "bucket");

-- CreateIndex
CREATE UNIQUE INDEX "Task_taskId_key" ON "Task"("taskId");

-- CreateIndex
CREATE INDEX "Task_taskId_idx" ON "Task"("taskId");

-- CreateIndex
CREATE INDEX "Task_shop_idx" ON "Task"("shop");

-- CreateIndex
CREATE INDEX "Task_status_idx" ON "Task"("status");

-- CreateIndex
CREATE INDEX "Task_createdAt_idx" ON "Task"("createdAt");

-- CreateIndex
CREATE INDEX "Task_completedAt_idx" ON "Task"("completedAt");

